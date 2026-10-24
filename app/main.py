import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.model_service import ChurnPredictor
from app.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from app.security import require_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("churn_api")

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load the model once, not per request.
    app.state.predictor = ChurnPredictor(settings.pipeline_path, settings.metadata_path)
    yield


app = FastAPI(
    title="ConnectTel Churn API",
    version="1.0.0",
    description="Predicts customer churn and explains each prediction with SHAP.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


def get_predictor(request: Request) -> ChurnPredictor:
    return request.app.state.predictor


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Liveness check used by Docker and the cloud platform."""
    loaded = getattr(request.app.state, "predictor", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get("/model/info", tags=["model"], dependencies=[Depends(require_api_key)])
def model_info(predictor: ChurnPredictor = Depends(get_predictor)) -> dict:
    """Training metrics, threshold and global feature importance."""
    return predictor.metadata


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["model"],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit)
def predict(
    request: Request,
    customer: CustomerFeatures,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictionResponse:
    result = predictor.predict(customer)
    logger.info("prediction probability=%.3f risk=%s", result.churn_probability, result.risk_level)
    return result
