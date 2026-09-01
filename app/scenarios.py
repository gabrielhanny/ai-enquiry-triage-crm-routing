"""Four synthetic demo scenarios, one per conditional path in the workflow."""
from .models import Channel, RawEnquiry

SCENARIOS: list[tuple[str, RawEnquiry]] = [
    (
        "Complete sales enquiry",
        RawEnquiry(
            channel=Channel.WEB_FORM,
            sender_name="Priya Nair",
            sender_email="priya.nair@northwind-retail.example",
            subject="Interested in your inventory platform",
            body=(
                "Hi, I'm Priya from Northwind Retail. We run 12 stores and are "
                "looking to replace our current inventory system this quarter. "
                "Could you send pricing for your Pro plan and set up a demo call?"
            ),
        ),
    ),
    (
        "Incomplete sales enquiry",
        RawEnquiry(
            channel=Channel.EMAIL,
            sender_name="Sam",
            sender_email=None,
            subject="pricing?",
            body="hey do you guys have pricing info, sounds interesting",
        ),
    ),
    (
        "Support enquiry",
        RawEnquiry(
            channel=Channel.MESSAGING,
            sender_name="Diego Alvarez",
            sender_email="diego.alvarez@bluecrest.example",
            subject=None,
            body=(
                "Our dashboard has been showing a 500 error since this morning "
                "whenever we try to export the monthly report (account bluecrest-042). "
                "Can someone take a look?"
            ),
        ),
    ),
    (
        "Junk enquiry",
        RawEnquiry(
            channel=Channel.EMAIL,
            sender_name="Winner Notice",
            sender_email="promo@totally-not-spam.example",
            subject="YOU HAVE WON A FREE PRIZE!!!",
            body="CLICK HERE NOW to claim your $1,000,000 reward!!! Limited time offer, act now!!!",
        ),
    ),
]
