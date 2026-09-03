from app import ai_queue
from app.models import Channel, RawEnquiry


def make_raw() -> RawEnquiry:
    return RawEnquiry(channel=Channel.EMAIL, sender_email="a@b.com", subject="hi", body="hello")


def test_enqueue_creates_a_queued_row():
    queue_id = ai_queue.enqueue(enquiry_id="e1", source_email_id="E001", raw=make_raw(), error="boom")
    items = ai_queue.list_all()
    assert len(items) == 1
    assert items[0]["id"] == queue_id
    assert items[0]["status"] == "queued"
    assert items[0]["source_email_id"] == "E001"
    assert items[0]["attempts"] == 1
    assert items[0]["last_error"] == "boom"


def test_enqueue_is_idempotent_on_source_email_id():
    first_id = ai_queue.enqueue(enquiry_id="e1", source_email_id="E001", raw=make_raw(), error="boom")
    second_id = ai_queue.enqueue(enquiry_id="e2", source_email_id="E001", raw=make_raw(), error="boom again")

    assert first_id == second_id
    items = ai_queue.list_all()
    assert len(items) == 1
    assert items[0]["attempts"] == 2
    assert items[0]["last_error"] == "boom again"


def test_enqueue_without_source_email_id_falls_back_to_enquiry_id():
    ai_queue.enqueue(enquiry_id="e1", source_email_id=None, raw=make_raw(), error="boom")
    ai_queue.enqueue(enquiry_id="e2", source_email_id=None, raw=make_raw(), error="boom")

    items = ai_queue.list_all()
    assert len(items) == 2  # different enquiry_id => different idempotency key


def test_mark_done_removes_item_from_queued_list():
    queue_id = ai_queue.enqueue(enquiry_id="e1", source_email_id="E001", raw=make_raw(), error="boom")
    assert len(ai_queue.list_queued()) == 1

    ai_queue.mark_done(queue_id)

    assert ai_queue.list_queued() == []
    assert ai_queue.list_all()[0]["status"] == "done"


def test_enqueue_does_not_resurrect_a_done_item():
    queue_id = ai_queue.enqueue(enquiry_id="e1", source_email_id="E001", raw=make_raw(), error="boom")
    ai_queue.mark_done(queue_id)

    ai_queue.enqueue(enquiry_id="e2", source_email_id="E001", raw=make_raw(), error="stray failure")

    items = ai_queue.list_all()
    assert len(items) == 1
    assert items[0]["status"] == "done"
    assert ai_queue.list_queued() == []


def test_payload_round_trips_the_raw_enquiry():
    raw = make_raw()
    ai_queue.enqueue(enquiry_id="e1", source_email_id="E001", raw=raw, error="boom")
    stored = ai_queue.list_all()[0]

    restored = RawEnquiry.model_validate_json(stored["payload"])
    assert restored.sender_email == raw.sender_email
    assert restored.body == raw.body
