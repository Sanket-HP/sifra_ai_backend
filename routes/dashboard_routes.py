from fastapi import APIRouter
from app.services.dashboard_with_insights import generate_dashboard_with_insights

router = APIRouter()

@router.get("/dashboard_with_insights")
def get_dashboard_with_insights():
    result = generate_dashboard_with_insights()
    return {"status": "success", "data": result}
