"""
OmniMemory FastAPI Server

REST API server providing access to all OmniMemory SDK functionality.
Uses lifespan management to initialize SDK once at startup.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Path, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from typing import Optional
from omnimemory.core.logger_utils import get_logger

from omnimemory.sdk import OmniMemorySDK
from omnimemory.core.schemas import (
    AddUserMessageRequest,
    ConversationSummaryRequest,
    ConversationSummaryResponse,
    AgentMemoryRequest,
    TaskResponse,
    MemoryResponse,
    MemoryListResponse,
    SuccessResponse,
)


logger = get_logger(name="omnimemory.api.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize and cleanup SDK."""
    logger.info("Initializing OmniMemorySDK...")
    try:
        app.state.sdk = OmniMemorySDK()
        warm_up_success = await app.state.sdk.warm_up()
        if warm_up_success:
            logger.info("VectorDB connection pool warm-up completed during startup")
        else:
            logger.warning("VectorDB connection pool warm-up failed during startup")
        logger.info("OmniMemorySDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OmniMemorySDK: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down OmniMemorySDK...")
    if hasattr(app.state, "sdk"):
        app.state.sdk = None


app = FastAPI(
    title="OmniMemory API",
    description="REST API for OmniMemory - Self-Evolving Composite Memory Synthesis Architecture (SECMSA)",
    version="0.1.0-beta",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_sdk(request: Request) -> OmniMemorySDK:
    """Get the SDK instance from app.state, raising error if not initialized."""
    if not hasattr(request.app.state, "sdk") or request.app.state.sdk is None:
        logger.error("SDK not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OmniMemorySDK not initialized",
        )
    return request.app.state.sdk


@app.post(
    "/api/v1/memories",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Add memory from user messages (async)",
    tags=["Memory Operations"],
)
async def add_memory(request: AddUserMessageRequest, http_request: Request):
    """
    Add memory from user messages and trigger asynchronous memory processing.

    Returns task information immediately. The memory is processed asynchronously.

    Validation: Messages count is validated by UserMessages schema (DEFAULT_MAX_MESSAGES).
    """
    try:
        logger.info(
            f"Adding memory: app_id={request.app_id}, user_id={request.user_id}, message_count={len(request.messages)}"
        )

        try:
            user_message = request.to_user_messages()
        except ValidationError as ve:
            error_details = []
            for error in ve.errors():
                field = "->".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field}: {error['msg']}")
            error_msg = "; ".join(error_details) if error_details else str(ve)
            logger.warning(f"Message validation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
            )
        except ValueError as ve:
            error_msg = str(ve)
            logger.warning(f"Message validation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
            )

        result = await get_sdk(http_request).add_memory(user_message)

        logger.info(f"Memory added successfully: task_id={result.get('task_id')}")
        return TaskResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add memory: {str(e)}",
        )


@app.post(
    "/api/v1/agent/summaries",
    tags=["Agent Operations"],
    summary="Generate a conversation summary via a single agent",
    responses={
        status.HTTP_200_OK: {"model": ConversationSummaryResponse},
        status.HTTP_202_ACCEPTED: {"model": TaskResponse},
    },
)
async def summarize_conversation(
    request: ConversationSummaryRequest, http_request: Request
):
    """
    Generate a conversation summary via a single agent.

    - If a callback URL is supplied, returns 202 Accepted immediately and
      delivers the summary to the webhook once ready.
    - Otherwise returns the summary payload synchronously.
    """
    try:
        sdk = get_sdk(http_request)
        result = await sdk.summarize_conversation(request)

        if isinstance(result, dict) and result.get("status") == "accepted":
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=result,
            )

        return ConversationSummaryResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error summarizing conversation: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize conversation: {str(exc)}",
        )


@app.post(
    "/api/v1/agent/memories",
    tags=["Agent Operations"],
    summary="Create memory from agent message (async)",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def add_agent_memory(request: AgentMemoryRequest, http_request: Request):
    """
    Create and store a memory from agent messages asynchronously.

    Returns task information immediately. The memory is processed asynchronously.

    Simple flow: agent sends messages (string or list), we generate a summary
    using the fast prompt, embed it, and store it directly. No conflict resolution,
    no linking, no metadata extraction - just store.
    """
    try:
        logger.info(
            f"Adding agent memory: app_id={request.app_id}, "
            f"user_id={request.user_id}, session_id={request.session_id}"
        )

        sdk = get_sdk(http_request)
        result = await sdk.add_agent_memory(request)

        logger.info(f"Agent memory task submitted: task_id={result.get('task_id')}")
        return TaskResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error adding agent memory: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add agent memory: {str(exc)}",
        )


@app.get(
    "/api/v1/memories/query",
    response_model=MemoryListResponse,
    summary="Query memories with intelligent ranking",
    tags=["Memory Operations"],
)
async def query_memory(
    http_request: Request,
    app_id: str = Query(..., description="Application ID"),
    query: str = Query(..., description="Natural language query", min_length=1),
    user_id: Optional[str] = Query(None, description="User ID filter"),
    session_id: Optional[str] = Query(None, description="Session ID filter"),
    n_results: Optional[int] = Query(
        None, description="Maximum number of results", ge=1, le=100
    ),
    similarity_threshold: Optional[float] = Query(
        None, description="Similarity threshold", ge=0.0, le=1.0
    ),
):
    """
    Query memory with intelligent multi-dimensional ranking.

    Performs semantic search with composite scoring combining:
    - Relevance (semantic similarity)
    - Recency (time-based freshness)
    - Importance (content significance)
    """
    try:
        logger.info(
            f"Querying memory: app_id={app_id}, query='{query[:50]}...', user_id={user_id}"
        )

        results = await get_sdk(http_request).query_memory(
            app_id=app_id,
            query=query,
            user_id=user_id,
            session_id=session_id,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
        )

        logger.info(f"Query completed: {len(results)} results returned")
        return MemoryListResponse(memories=results, count=len(results))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query memory: {str(e)}",
        )


@app.get(
    "/api/v1/memories/{memory_id}",
    response_model=MemoryResponse,
    summary="Get a single memory by ID",
    tags=["Memory Operations"],
)
async def get_memory(
    http_request: Request,
    memory_id: str = Path(..., description="Memory ID"),
    app_id: str = Query(..., description="Application ID"),
):
    """
    Get a single memory by its ID.

    Returns the complete memory data including document, metadata, and all fields.
    """
    try:
        logger.info(f"Getting memory: memory_id={memory_id}, app_id={app_id}")

        memory = await get_sdk(http_request).get_memory(
            memory_id=memory_id, app_id=app_id
        )

        if memory is None:
            logger.warning(f"Memory not found: memory_id={memory_id}, app_id={app_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found in app {app_id}",
            )

        logger.info(f"Memory retrieved successfully: memory_id={memory_id}")
        return MemoryResponse(**memory)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory: {str(e)}",
        )


@app.delete(
    "/api/v1/memories/{memory_id}",
    response_model=SuccessResponse,
    summary="Delete a memory",
    tags=["Memory Operations"],
)
async def delete_memory(
    http_request: Request,
    memory_id: str = Path(..., description="Memory ID"),
    app_id: str = Query(..., description="Application ID"),
):
    """
    Delete a memory from the collection.

    Returns success status.
    """
    try:
        logger.info(f"Deleting memory: memory_id={memory_id}, app_id={app_id}")

        success = await get_sdk(http_request).delete_memory(
            app_id=app_id, doc_id=memory_id
        )

        if success:
            logger.info(f"Memory deleted successfully: memory_id={memory_id}")
            return SuccessResponse(
                success=True, message=f"Memory {memory_id} deleted successfully"
            )
        else:
            logger.warning(f"Failed to delete memory: memory_id={memory_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete memory {memory_id}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}",
        )


@app.get(
    "/api/v1/memories/{memory_id}/evolution",
    response_model=MemoryListResponse,
    summary="Traverse memory evolution chain",
    tags=["Memory Operations"],
)
async def traverse_memory_evolution_chain(
    http_request: Request,
    memory_id: str = Path(..., description="Starting memory ID"),
    app_id: str = Query(..., description="Application ID"),
):
    """
    Traverse the memory evolution chain using singly linked list algorithm.

    Starting from the given memory_id, follows the next_id links forward until
    reaching None, collecting all memories in the evolution chain.

    Returns memories in evolution order (oldest to newest).
    """
    try:
        logger.info(
            f"Traversing memory evolution chain: memory_id={memory_id}, app_id={app_id}"
        )

        chain = await get_sdk(http_request).traverse_memory_evolution_chain(
            app_id=app_id,
            memory_id=memory_id,
        )

        logger.info(
            f"Memory evolution chain traversal completed: {len(chain)} memories found"
        )
        return MemoryListResponse(memories=chain, count=len(chain))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error traversing memory evolution chain: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to traverse memory evolution chain: {str(e)}",
        )


@app.get(
    "/api/v1/memories/{memory_id}/evolution/graph",
    summary="Generate graph visualization of memory evolution chain",
    tags=["Memory Operations"],
)
async def generate_evolution_graph(
    http_request: Request,
    memory_id: str = Path(..., description="Starting memory ID"),
    app_id: str = Query(..., description="Application ID"),
    format: str = Query("mermaid", description="Graph format: mermaid, dot, or html"),
):
    """
    Generate a graph visualization of the memory evolution chain.

    Supports multiple formats:
    - mermaid: Mermaid diagram syntax (text-based, widely supported)
    - dot: Graphviz DOT format (can be rendered to PNG/SVG)
    - html: HTML file with embedded Mermaid.js visualization
    """
    try:
        logger.info(
            f"Generating evolution graph: memory_id={memory_id}, app_id={app_id}, format={format}"
        )

        chain = await get_sdk(http_request).traverse_memory_evolution_chain(
            app_id=app_id,
            memory_id=memory_id,
        )

        if not chain:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No evolution chain found starting from {memory_id}",
            )

        graph_output = get_sdk(http_request).generate_evolution_graph(
            chain=chain, format=format
        )

        if not graph_output:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate graph",
            )

        if format == "html":
            from fastapi.responses import HTMLResponse

            return HTMLResponse(content=graph_output)
        else:
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse(content=graph_output)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating evolution graph: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate evolution graph: {str(e)}",
        )


@app.get("/health", summary="Health check", tags=["System"])
async def health_check(http_request: Request):
    """Health check endpoint."""
    sdk_initialized = (
        hasattr(http_request.app.state, "sdk")
        and http_request.app.state.sdk is not None
    )
    return {
        "status": "healthy",
        "sdk_initialized": sdk_initialized,
        "service": "omnimemory-api",
    }


@app.get(
    "/api/v1/system/pool-stats",
    summary="Get vector DB connection pool stats",
    tags=["System"],
)
async def get_pool_stats(http_request: Request):
    """Expose connection pool statistics for observability."""
    try:
        stats = await get_sdk(http_request).get_connection_pool_stats()
        return stats
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to fetch pool stats: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch pool stats: {exc}",
        )


@app.get("/", summary="API information", tags=["System"])
async def root():
    """API root endpoint with basic information."""
    return {
        "name": "OmniMemory API",
        "version": "0.1.0-beta",
        "architecture": "Self-Evolving Composite Memory Synthesis Architecture (SECMSA)",
        "description": "REST API for OmniMemory - Dual-Agent Construction, Persistence Storage, Self-Evolution",
        "endpoints": {"docs": "/docs", "health": "/health", "api": "/api/v1"},
    }
