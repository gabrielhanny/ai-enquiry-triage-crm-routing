from app.ingest import (
    email_to_raw_enquiry,
    load_crm_records,
    load_emails,
    load_staff_directory,
    parse_from_header,
)
from app.models import EmailMessage


def test_load_staff_directory_reads_all_four_members():
    staff = load_staff_directory()
    assert len(staff) == 4
    assert {member.name for member in staff} == {
        "Matt Cooper",
        "Ties Rahardjo",
        "Zidane Mouldino",
        "Ali Pratama",
    }


def test_load_crm_records_reads_all_five_rows():
    records = load_crm_records()
    assert len(records) == 5
    assert records[0].id == "C001"
    # C002's phone is blank in the fixture — preserved verbatim, not invented.
    c002 = next(r for r in records if r.id == "C002")
    assert c002.phone == ""


def test_load_emails_reads_all_twelve():
    emails = load_emails()
    assert len(emails) == 12
    assert emails[0].id == "E001"
    assert emails[0].attachment == "01_hume_energy_bill.txt"


def test_parse_from_header_handles_name_and_email():
    name, email = parse_from_header("Amelia Grant <amelia.grant@humelogistics.example>")
    assert name == "Amelia Grant"
    assert email == "amelia.grant@humelogistics.example"


def test_parse_from_header_handles_bare_email():
    name, email = parse_from_header("a.grant@humelogistics.example")
    assert name is None
    assert email == "a.grant@humelogistics.example"


def test_email_to_raw_enquiry_combines_body_and_attachment_as_evidence():
    email = EmailMessage(
        id="E001",
        **{"from": "Amelia Grant <amelia.grant@humelogistics.example>"},
        subject="Solar and battery across our three Victorian sites",
        body="We operate warehouses in Truganina, Dandenong and Epping.",
        attachment="01_hume_energy_bill.txt",
    )
    raw = email_to_raw_enquiry(email)
    assert raw.sender_email == "amelia.grant@humelogistics.example"
    assert raw.sender_name == "Amelia Grant"
    assert "Truganina" in raw.body
    assert "01_hume_energy_bill.txt" in raw.body
    assert "Total bill: $18,940" in raw.body  # attachment content pulled in verbatim


def test_email_to_raw_enquiry_without_attachment_is_unaffected():
    email = EmailMessage(id="E004", **{"from": "sales@megaleadlists.example"}, subject="x", body="Buy leads now.")
    raw = email_to_raw_enquiry(email)
    assert raw.body == "Buy leads now."
