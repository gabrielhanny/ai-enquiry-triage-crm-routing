from app.ingest import load_staff_directory
from app.staff import recommend_owner


def test_sales_routes_to_founder_via_commercial_keyword():
    staff = load_staff_directory()
    rec = recommend_owner("sales", staff)
    assert rec.owner == "Matt Cooper"


def test_infrastructure_routes_to_business_analyst():
    staff = load_staff_directory()
    rec = recommend_owner("infrastructure", staff)
    assert rec.owner == "Ali Pratama"


def test_operations_routes_to_operations_coordinator():
    staff = load_staff_directory()
    rec = recommend_owner("operations", staff)
    assert rec.owner == "Ties Rahardjo"


def test_unassigned_when_no_staff_directory_supplied():
    rec = recommend_owner("sales", [])
    assert rec.owner is None
    assert rec.confidence == 0.0


def test_junk_has_no_owner_recommendation():
    staff = load_staff_directory()
    rec = recommend_owner("junk", staff)
    assert rec.owner is None
