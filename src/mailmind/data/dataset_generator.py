"""Synthetic, labelled email-corpus generator for MailMind AI.

This module fabricates a rich, realistic and *deterministic* training corpus for
the six target categories (``Important``, ``Work``, ``Personal``, ``Social``,
``Promotions`` and ``Spam``). Every category is backed by a dozen-plus subject
and body templates filled from curated slot vocabularies (names, companies,
products, amounts, dates), so the rows are varied rather than near-duplicates
while still carrying strong, separable per-category signals.

The corpus is the single source of truth for model training and evaluation, so
generation is fully reproducible: given the same ``seed`` it always yields the
same frame. Rows are assembled through :class:`mailmind.schema.Email` so that
the ``id`` and ``sender_domain`` fields are derived exactly as the rest of the
pipeline expects.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from mailmind import config
from mailmind.schema import Email

# --------------------------------------------------------------------------- #
# Fixed reference point so timestamps are deterministic and bounded to the
# "last 30 days" relative to a frozen "now".
# --------------------------------------------------------------------------- #
_REFERENCE_NOW: datetime = datetime(2026, 6, 14, 9, 0, 0)
_MAX_AGE_MINUTES: int = 30 * 24 * 60  # 30 days expressed in minutes

# Exact output column order required by the dataset contract.
_COLUMNS: list[str] = [
    "id",
    "sender",
    "sender_name",
    "sender_domain",
    "subject",
    "body",
    "timestamp",
    "has_attachment",
    "num_links",
    "label",
]

# --------------------------------------------------------------------------- #
# Shared slot vocabularies
# --------------------------------------------------------------------------- #
_FIRST_NAMES: list[str] = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Isla", "James", "Kira", "Liam", "Maya", "Noah", "Olivia", "Priya",
    "Quentin", "Rohan", "Sara", "Tom", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane",
]
_LAST_NAMES: list[str] = [
    "Smith", "Johnson", "Patel", "Garcia", "Nguyen", "Brown", "Khan", "Lee",
    "Martinez", "Wright", "Kumar", "Davis", "Lopez", "Singh", "Clark",
    "Rossi", "Chen", "Murphy", "Ali", "Walsh",
]
_COMPANIES: list[str] = [
    "Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Soylent",
    "Vandelay", "Stark Industries", "Wayne Enterprises", "Wonka Industries",
    "Cyberdyne", "Massive Dynamic",
]
_PRODUCTS: list[str] = [
    "wireless earbuds", "running shoes", "smart watch", "coffee maker",
    "yoga mat", "backpack", "office chair", "desk lamp", "phone case",
    "blender", "sunglasses", "water bottle",
]
_MONTHS: list[str] = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
_WEEKDAYS: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
]
_CITIES: list[str] = [
    "Goa", "Lisbon", "Bali", "Kyoto", "Austin", "Porto", "Seville", "Hanoi",
]


# --------------------------------------------------------------------------- #
# Small slot-filling helpers
# --------------------------------------------------------------------------- #
def _full_name(rng: random.Random) -> str:
    """Return a random ``First Last`` display name."""
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _username(name: str, rng: random.Random) -> str:
    """Derive a plausible mailbox local-part from a display name."""
    first, _, last = name.lower().partition(" ")
    patterns = [
        f"{first}.{last}",
        f"{first}{last}",
        f"{first}.{last[:1]}",
        f"{first[:1]}{last}",
        f"{first}{rng.randint(1, 99)}",
    ]
    return rng.choice(patterns)


def _amount(rng: random.Random) -> str:
    """Return a formatted currency amount string."""
    symbol = rng.choice(["$", "€", "£", "₹"])
    value = rng.choice(
        [rng.randint(9, 99), rng.randint(100, 999), rng.randint(1000, 9999)]
    )
    return f"{symbol}{value:,}"


def _date_phrase(rng: random.Random) -> str:
    """Return a human date phrase such as ``June 21``."""
    return f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}"


def _order_no(rng: random.Random) -> str:
    """Return a random order/reference number."""
    return f"#{rng.randint(100000, 999999)}"


def _percent(rng: random.Random) -> int:
    """Return a marketing discount percentage."""
    return rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 70])


def _otp(rng: random.Random) -> int:
    """Return a six-digit one-time passcode."""
    return rng.randint(100000, 999999)


# --------------------------------------------------------------------------- #
# Per-category template banks
# --------------------------------------------------------------------------- #
# Each ``_*_TEMPLATES`` constant is a ``(subjects, bodies)`` pair. Subjects and
# bodies are independent template lists (>= 12 each) so that random pairing
# yields a large variety of realistic combinations per category. Templates use
# ``str.format`` style slots filled by :func:`_fill`.

_IMPORTANT_SUBJECTS: list[str] = [
    "Action required: invoice {order} is due",
    "Payment due for invoice {order}",
    "Security alert: new sign-in to your account",
    "Your password reset request",
    "Verification code: {otp}",
    "Urgent: contract awaiting your approval",
    "Final notice: account suspension pending",
    "Your {company} account statement is ready",
    "Important: tax document for {month}",
    "Approval needed before {date}",
    "Account security: unusual activity detected",
    "Your {amount} payment was declined",
    "Legal notice regarding your account",
    "Two-factor authentication code: {otp}",
]
_IMPORTANT_BODIES: list[str] = [
    "Invoice {order} for {amount} is due on {date}. Please complete payment to "
    "avoid a late fee.",
    "We detected a sign-in to your account from a new device. If this was not "
    "you, reset your password immediately.",
    "Use the verification code {otp} to confirm your identity. This code "
    "expires in 10 minutes. Do not share it with anyone.",
    "Your password reset link is ready. For your security it will expire "
    "within 30 minutes. If you did not request this, contact support.",
    "The {company} contract requires your signature before {date}. Please "
    "review the attached document and approve at your earliest convenience.",
    "This is a final notice: your account will be suspended on {date} unless "
    "the outstanding balance of {amount} is cleared.",
    "Your monthly account statement for {month} is now available. The closing "
    "balance is {amount}.",
    "Action is required to keep your account active. Please verify your "
    "billing details before {date}.",
    "We have flagged unusual activity on your account. Please review the "
    "recent transactions totalling {amount} and confirm they are yours.",
    "Your payment of {amount} could not be processed. Update your payment "
    "method to avoid interruption of service.",
    "Please find the tax document for {month} attached. Retain this for your "
    "records and submit before the filing deadline.",
    "Your one-time passcode is {otp}. Enter it to complete the secure "
    "transaction for invoice {order}.",
    "The HR team needs your approval on the updated agreement by {date}. This "
    "is time sensitive and cannot be deferred.",
    "A legal notice has been issued regarding your account. Review the "
    "attached letter and respond before {date}.",
]

_WORK_SUBJECTS: list[str] = [
    "Project {company}: status update",
    "Meeting agenda for {weekday}",
    "Sprint planning notes",
    "Standup summary - {date}",
    "Quarterly report attached",
    "Review request: {product} spec",
    "Client call recap - {company}",
    "Deliverable due {date}",
    "Schedule for next week",
    "Re: deployment plan",
    "Design review for the new feature",
    "Action items from today's sync",
    "Budget proposal for {company}",
    "Weekly status: {company} account",
]
_WORK_BODIES: list[str] = [
    "Hi team, here is the status update for the {company} project. We are on "
    "track for the {date} milestone; let me know if you have blockers.",
    "Please review the agenda for our {weekday} meeting. We will cover the "
    "roadmap, open risks, and the {product} timeline.",
    "Sprint planning notes are attached. Each story has been pointed; please "
    "confirm your assignments before {date}.",
    "Quick standup summary: backend is unblocked, the {product} integration is "
    "in review, and QA starts {weekday}.",
    "The quarterly report is attached. Revenue is up and the {company} account "
    "renewed. Happy to walk through the numbers.",
    "Could you review the {product} spec by {date}? I left comments in the "
    "doc and would value your feedback before we ship.",
    "Recap of the client call with {company}: they approved scope, want a demo "
    "by {date}, and asked about pricing tiers.",
    "Reminder that the {product} deliverable is due {date}. Please push your "
    "branch and request review so we can merge in time.",
    "Here is the schedule for next week. The {company} review is on {weekday}; "
    "the retro follows in the afternoon.",
    "Following up on the deployment plan. We will release on {weekday} during "
    "the maintenance window; rollback steps are in the runbook.",
    "Design review for the new feature is set for {weekday}. Please come with "
    "feedback on the {product} mockups attached here.",
    "Action items from today's sync: finalise the {company} estimate, update "
    "the tracker, and schedule the demo for {date}.",
    "Attaching the budget proposal for the {company} engagement. Numbers need "
    "sign-off before {date} so finance can process it.",
    "Hi {first}, can you take the review on my pull request today? It blocks "
    "the {product} release planned for {date}.",
]

_PERSONAL_SUBJECTS: list[str] = [
    "Dinner this {weekday}?",
    "Weekend plans",
    "Happy birthday!",
    "Photos from {city}",
    "Long time no see - catch up?",
    "Vacation in {city}",
    "Mom's visit next month",
    "Movie night?",
    "Thanks for the other day",
    "Quick question",
    "Are you free {date}?",
    "Family dinner on {weekday}",
    "Miss you!",
    "Road trip idea",
]
_PERSONAL_BODIES: list[str] = [
    "Hey {first}, are you free for dinner this {weekday}? Thinking of trying "
    "that new place downtown. Let me know!",
    "What are you up to this weekend? We were thinking of a hike and maybe a "
    "barbecue after. You in?",
    "Happy birthday!! Hope you have an amazing day. Let's celebrate properly "
    "when I'm back from {city}. Miss you lots.",
    "Finally uploaded the photos from {city} - they came out great! Sending a "
    "few of my favourites. That trip was so much fun.",
    "It's been way too long! Want to catch up over coffee on {date}? So much "
    "has happened, I want to hear everything.",
    "We booked the {city} trip for next month! Can't wait. Do you still have "
    "that travel guide you mentioned?",
    "Mom is visiting next month and would love to see you. Are you around on "
    "{weekday}? She keeps asking about you.",
    "Movie night at mine on {weekday}? I'll get snacks, you pick the film. "
    "Bring {second} along if they're free.",
    "Just wanted to say thanks for the other day, it really meant a lot. Let "
    "me return the favour soon - dinner's on me.",
    "Quick one - do you still have my charger? Think I left it at yours after "
    "the party. No rush, just wondering.",
    "Are you free on {date}? A few of us are getting together and it wouldn't "
    "be the same without you. Hope you can make it!",
    "Family dinner is on {weekday} at 7. Grandma's cooking, so you know it'll "
    "be a feast. Let me know if you can bring dessert.",
    "Miss you! Feels like ages since we properly hung out. Free for a call "
    "this week to catch up properly?",
    "Random idea: road trip to {city} next long weekend? We could split the "
    "driving. Tell me you're in!",
]

_SOCIAL_SUBJECTS: list[str] = [
    "{first} liked your post",
    "You have a new follower",
    "{first} commented on your photo",
    "{first} sent you a friend request",
    "{first} mentioned you in a comment",
    "You were tagged in a photo",
    "{first} wants to connect on LinkedIn",
    "New notifications from your network",
    "{first} started following you",
    "Your post is trending",
    "{first} reacted to your story",
    "People you may know",
    "{first} invited you to an event",
    "You have 5 new notifications",
]
_SOCIAL_BODIES: list[str] = [
    "{first} {last} and 12 others liked your recent post. See who else "
    "engaged with your update on the app.",
    "You have a new follower! {first} {last} just started following you. "
    "Check out their profile and follow back.",
    "{first} {last} commented on your photo: \"Looks amazing!\" Tap to reply "
    "and keep the conversation going.",
    "{first} {last} sent you a friend request. You have 3 mutual friends. "
    "Accept to connect and see their updates.",
    "{first} {last} mentioned you in a comment. Open the app to see the full "
    "thread and respond.",
    "You were tagged in a photo by {first} {last}. Review the tag and choose "
    "whether it appears on your timeline.",
    "{first} {last} would like to connect on LinkedIn. They work at "
    "{company}. Accept the invitation to grow your network.",
    "You have new notifications from your network this week, including likes, "
    "comments, and {count} new connection requests.",
    "{first} {last} started following you and reacted to 4 of your posts. See "
    "their activity in your notifications.",
    "Your post is trending! It has reached {count} people and is getting more "
    "engagement than 90% of your posts.",
    "{first} {last} reacted to your story. Stories disappear in 24 hours - "
    "open the app to see who's watching.",
    "People you may know: {first} {last} and {count} others. Connect now to "
    "expand your professional circle.",
    "{first} {last} invited you to the event \"{company} Meetup\". Let them "
    "know if you can attend by responding in the app.",
    "You have {count} new notifications, including comments on your photo and "
    "a new message request from {first} {last}.",
]

_PROMO_SUBJECTS: list[str] = [
    "{percent}% off everything - today only!",
    "Flash sale: {product} now {amount}",
    "Your exclusive coupon inside",
    "Free shipping on all orders this week",
    "Limited time: {percent}% off {product}",
    "New arrivals you'll love",
    "Last chance: sale ends tonight",
    "{first}, a special deal just for you",
    "Weekend deals up to {percent}% off",
    "Our biggest sale of the season",
    "Don't miss {percent}% off sitewide",
    "Your cart is waiting - {percent}% off",
    "The {month} newsletter is here",
    "Buy one get one free on {product}",
]
_PROMO_BODIES: list[str] = [
    "Hurry! Get {percent}% off everything for today only. Use code SAVE{percent} "
    "at checkout. Shop now before the deal ends. Unsubscribe anytime.",
    "Our flash sale is live: grab the {product} for just {amount}, down from "
    "full price. Free shipping included. Visit the link to shop.",
    "Here's your exclusive coupon for {percent}% off your next order. Valid "
    "until {date}. Click the link to redeem. To opt out, unsubscribe below.",
    "Enjoy free shipping on all orders this week - no minimum required. Browse "
    "the new collection and treat yourself. Unsubscribe link at the bottom.",
    "Limited time offer: {percent}% off the {product} everyone's talking "
    "about. Tap to shop the deal before stock runs out.",
    "New arrivals just dropped! Discover fresh styles picked for you and enjoy "
    "{percent}% off your first order. Shop the collection now.",
    "Last chance - our sale ends tonight at midnight. Save up to {percent}% "
    "across the store. Don't miss out. Click to browse the deals.",
    "Hi {first}, we picked a special deal just for you: {percent}% off the "
    "{product}. Redeem with the link below. Unsubscribe to stop these emails.",
    "Weekend deals are here with up to {percent}% off select items. Your "
    "coupon SAVE{percent} is ready. Shop now while supplies last.",
    "Our biggest sale of the season starts now. Save {percent}% on bestsellers "
    "including the {product}. Visit the link to start saving today.",
    "Don't miss {percent}% off sitewide this week only. Add your favourites to "
    "the cart and use code DEAL{percent}. Free returns on everything.",
    "You left the {product} in your cart! Complete your order now and take "
    "{percent}% off as a thank-you. Click the link to check out.",
    "The {month} newsletter is here with new drops, styling tips, and a "
    "subscriber-only {percent}% discount. Read more via the link below.",
    "Buy one get one free on the {product} this week. Stock up and save - just "
    "add two to your cart. Unsubscribe link below if you'd prefer fewer emails.",
]

_SPAM_SUBJECTS: list[str] = [
    "CONGRATULATIONS!!! You WON {amount}!!!",
    "Claim your prize NOW before it expires",
    "URGENT: verify your account immediately",
    "You are today's lucky winner!!!",
    "Make {amount} per day from home!!!",
    "Your account has been SUSPENDED - act now",
    "Hot crypto tip: 1000% returns guaranteed",
    "FINAL WARNING: respond within 24 hours",
    "You've been selected for a free {product}",
    "Miracle cure doctors don't want you to know",
    "Wire transfer pending - confirm details",
    "Lottery winner notification!!!",
    "Re: your inheritance of {amount}",
    "CLICK HERE to unlock your reward!!!",
]
_SPAM_BODIES: list[str] = [
    "CONGRATULATIONS!!! You have been selected to receive {amount}!!! To CLAIM "
    "your prize, click the link and confirm your bank details NOW!!!",
    "You are today's LUCKY WINNER!!! Reply with your full name and account "
    "number to release your reward of {amount}. ACT FAST - offer expires soon!",
    "URGENT: your account has been compromised. Verify your password and "
    "credit card immediately by clicking this link or it will be DELETED!!!",
    "Earn {amount} per day working from home!!! No experience needed!!! Join "
    "thousands already getting rich. Click the link to start TODAY!!!",
    "Your account has been SUSPENDED due to suspicious activity. Confirm your "
    "login and SSN within 24 hours or lose access FOREVER. Click here NOW!!!",
    "EXCLUSIVE crypto opportunity!!! Invest now and earn 1000% guaranteed "
    "returns!!! Send your wallet seed phrase to secure your spot. Don't wait!!!",
    "FINAL WARNING!!! You must respond within 24 hours to avoid legal action. "
    "Wire {amount} to the account below to settle immediately. Click here.",
    "Dear winner, you have won the international lottery!!! To collect your "
    "{amount} prize, send a processing fee and your passport copy ASAP!!!",
    "You've been SPECIALLY selected for a FREE {product}!!! Just pay shipping "
    "and provide your card number. CLICK NOW before this offer disappears!!!",
    "Doctors HATE this!!! This one miracle cure melts fat overnight!!! Order "
    "now at a secret discount - limited supply. Click the link immediately!!!",
    "A wire transfer of {amount} is pending in your name. Confirm your banking "
    "details and password through the secure link to release the funds NOW!!!",
    "Greetings, I am a foreign official with an inheritance of {amount} to "
    "transfer to you. Send your details and a small fee to proceed urgently.",
    "Your package could not be delivered!!! Pay the {amount} customs fee via "
    "the link and verify your card to reschedule delivery. URGENT action!!!",
    "CLICK HERE to unlock your reward of {amount}!!! This is NOT a scam!!! Just "
    "verify your identity and bank PIN to receive your money instantly!!!",
]

# Map each category to its (subjects, bodies) template bank.
_TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "Important": (_IMPORTANT_SUBJECTS, _IMPORTANT_BODIES),
    "Work": (_WORK_SUBJECTS, _WORK_BODIES),
    "Personal": (_PERSONAL_SUBJECTS, _PERSONAL_BODIES),
    "Social": (_SOCIAL_SUBJECTS, _SOCIAL_BODIES),
    "Promotions": (_PROMO_SUBJECTS, _PROMO_BODIES),
    "Spam": (_SPAM_SUBJECTS, _SPAM_BODIES),
}

# Realistic confusion structure: which categories a given category is most often
# mistaken for. Used to fabricate "hard" examples and (optionally) label noise so
# the corpus is not perfectly separable. Ordered most- to least-confusable.
_CONFUSABLE: dict[str, list[str]] = {
    "Important": ["Work", "Spam"],
    "Work": ["Important", "Personal"],
    "Personal": ["Social", "Work"],
    "Social": ["Promotions", "Personal"],
    "Promotions": ["Spam", "Social"],
    "Spam": ["Promotions", "Important"],
}

# --------------------------------------------------------------------------- #
# Per-category sender domain pools
# --------------------------------------------------------------------------- #
_DOMAINS: dict[str, list[str]] = {
    "Important": [
        "bank.com", "gov.in", "university.edu", "hr.company.com",
        "finance.company.com", "secure-billing.com", "payments.com",
        "accounts.company.com",
    ],
    "Work": [
        "work.com", "company.com", "acme-corp.com", "globex.com",
        "initech.com", "team.company.com",
    ],
    "Personal": [
        "gmail.com", "outlook.com", "yahoo.com", "icloud.com", "hotmail.com",
    ],
    "Social": [
        "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
        "notifications.linkedin.com", "mail.instagram.com",
    ],
    "Promotions": [
        "deals.shop.com", "newsletter.retail.com", "promo.store.com",
        "offers.brand.com", "shop.com", "marketing.bigsale.com",
    ],
    "Spam": [
        "win-big-now.biz", "claim-prize.info", "secure-verify.xyz",
        "lucky-lotto.top", "crypto-gains.click", "free-reward.online",
    ],
}

# Display-name strategy per category: some senders are people, some are brands.
_BRAND_NAMES: dict[str, list[str]] = {
    "Important": [
        "Account Security", "Billing Team", "HR Department", "Finance Team",
        "Payments", "University Registrar",
    ],
    "Social": [
        "LinkedIn", "Facebook", "Twitter", "Instagram", "Your Network",
    ],
    "Promotions": [
        "Shop Deals", "The Store", "Retail Newsletter", "Big Sale",
        "Brand Offers",
    ],
    "Spam": [
        "Prize Department", "Lucky Winner", "Account Verification",
        "Crypto Team", "Lottery Office",
    ],
}


# --------------------------------------------------------------------------- #
# Slot filling
# --------------------------------------------------------------------------- #
def _fill(template: str, rng: random.Random, recipient_first: str) -> str:
    """Fill every supported ``{slot}`` in *template* using *rng*.

    Only the slots actually present in the template are computed, keeping
    generation cheap and deterministic. The ``{first}``/``{last}`` slots refer
    to a *third party* mentioned in the message, while ``{recipient}`` is the
    name of the person being written to.
    """
    name = _full_name(rng)
    first, _, last = name.partition(" ")
    values: dict[str, str] = {
        "first": first,
        "last": last,
        "second": rng.choice(_FIRST_NAMES),
        "recipient": recipient_first,
        "company": rng.choice(_COMPANIES),
        "product": rng.choice(_PRODUCTS),
        "month": rng.choice(_MONTHS),
        "weekday": rng.choice(_WEEKDAYS),
        "city": rng.choice(_CITIES),
        "amount": _amount(rng),
        "date": _date_phrase(rng),
        "order": _order_no(rng),
        "percent": str(_percent(rng)),
        "otp": str(_otp(rng)),
        "count": str(rng.randint(2, 48)),
    }
    return template.format_map(values)


# --------------------------------------------------------------------------- #
# Per-category metadata generators (attachments / links)
# --------------------------------------------------------------------------- #
def _attachment_prob(category: str) -> float:
    """Probability that a message in *category* carries an attachment."""
    return {
        "Important": 0.55,
        "Work": 0.60,
        "Personal": 0.18,
        "Social": 0.02,
        "Promotions": 0.05,
        "Spam": 0.10,
    }.get(category, 0.1)


def _num_links(category: str, rng: random.Random) -> int:
    """Sample a realistic link count for *category*."""
    if category == "Promotions":
        return rng.randint(2, 6)
    if category == "Spam":
        return rng.randint(1, 5)
    if category == "Social":
        return rng.randint(1, 3)
    if category == "Important":
        return rng.randint(0, 2)
    if category == "Work":
        return rng.randint(0, 2)
    return rng.randint(0, 1)  # Personal


def _sender_for(category: str, rng: random.Random) -> tuple[str, str, str]:
    """Return a ``(sender, sender_name, domain)`` triple for *category*.

    People-style categories use a personal display name and a matching mailbox;
    brand-style categories use a brand display name and a generic mailbox.
    """
    domain = rng.choice(_DOMAINS[category])
    brands = _BRAND_NAMES.get(category)
    if brands is not None and rng.random() < 0.85:
        sender_name = rng.choice(brands)
        local = rng.choice(
            ["no-reply", "notifications", "alerts", "team", "info", "support"]
        )
    else:
        sender_name = _full_name(rng)
        local = _username(sender_name, rng)
    sender = f"{local}@{domain}"
    return sender, sender_name, domain


def _timestamp(rng: random.Random) -> str:
    """Return an ISO-8601 timestamp within the last 30 days of the reference."""
    minutes_ago = rng.randint(0, _MAX_AGE_MINUTES)
    when = _REFERENCE_NOW - timedelta(minutes=minutes_ago)
    return when.isoformat()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_dataset(
    samples_per_category: int = config.SAMPLES_PER_CATEGORY,
    seed: int = config.RANDOM_SEED,
    ambiguity: float = config.DATASET_AMBIGUITY,
    label_noise: float = config.DATASET_LABEL_NOISE,
) -> pd.DataFrame:
    """Generate a balanced, labelled synthetic email corpus.

    Args:
        samples_per_category: Number of rows to emit for each of the six
            categories in :data:`mailmind.config.CATEGORIES`.
        seed: Seed for the local PRNG; identical seeds yield identical frames.
        ambiguity: Fraction (0-1) of rows made deliberately *hard* — their body
            is drawn from a confusable neighbouring category while the label and
            sender metadata stay true to the original category. This injects the
            realistic overlap real inboxes show, so the model is not handed a
            perfectly separable problem. Set to ``0`` for clean, separable data.
        label_noise: Fraction (0-1) of rows whose *label* is flipped to a
            confusable neighbour, modelling annotator error. Defaults to ``0``
            (clean ground truth); the training pipeline raises it slightly.

    Returns:
        A :class:`pandas.DataFrame` with exactly the columns (in order)::

            id, sender, sender_name, sender_domain, subject, body,
            timestamp, has_attachment, num_links, label

        and ``samples_per_category * len(CATEGORIES)`` rows. When
        ``label_noise`` is ``0`` the classes remain perfectly balanced.
    """
    rng = random.Random(seed)
    rows: list[dict] = []

    for category in config.CATEGORIES:
        att_prob = _attachment_prob(category)
        neighbours = _CONFUSABLE[category]
        for _ in range(samples_per_category):
            recipient_first = rng.choice(_FIRST_NAMES)

            # Decide whether this row is a "hard" / ambiguous example whose body
            # mimics a confusable neighbour (subject mimics it half the time).
            is_hard = rng.random() < ambiguity
            subj_cat = category
            body_cat = category
            if is_hard:
                neighbour = rng.choice(neighbours)
                body_cat = neighbour
                if rng.random() < 0.5:
                    subj_cat = neighbour

            subject = _fill(rng.choice(_TEMPLATES[subj_cat][0]), rng, recipient_first)
            body = _fill(rng.choice(_TEMPLATES[body_cat][1]), rng, recipient_first)
            sender, sender_name, domain = _sender_for(category, rng)

            # Optional annotator-error label noise (off by default).
            label = category
            if rng.random() < label_noise:
                label = rng.choice(neighbours)

            email = Email(
                subject=subject,
                body=body,
                sender=sender,
                sender_name=sender_name,
                sender_domain=domain,
                recipient=f"{recipient_first.lower()}@gmail.com",
                timestamp=_timestamp(rng),
                has_attachment=rng.random() < att_prob,
                num_links=_num_links(category, rng),
                label=label,
            )
            record = email.to_dict()
            record["label"] = label
            rows.append({col: record[col] for col in _COLUMNS})

    frame = pd.DataFrame(rows, columns=_COLUMNS)
    return frame


def save_dataset(df: pd.DataFrame, path=config.DATASET_PATH) -> None:
    """Write *df* to *path* as CSV (no index), creating parent dirs as needed."""
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path=config.DATASET_PATH) -> pd.DataFrame:
    """Load a previously saved dataset CSV into a :class:`pandas.DataFrame`."""
    return pd.read_csv(_as_path(path))


def dataset_summary(df: pd.DataFrame) -> str:
    """Return a human-readable per-category count summary of *df*."""
    counts = df["label"].value_counts()
    width = max((len(c) for c in config.CATEGORIES), default=10)
    lines = ["Dataset summary", "=" * (width + 12)]
    for category in config.CATEGORIES:
        lines.append(f"{category:<{width}} : {int(counts.get(category, 0)):>6}")
    lines.append("-" * (width + 12))
    lines.append(f"{'TOTAL':<{width}} : {len(df):>6}")
    return "\n".join(lines)


def _as_path(path) -> Path:
    """Coerce *path* to a :class:`pathlib.Path` (idempotent for Path inputs)."""
    return path if isinstance(path, Path) else Path(path)


def main() -> None:
    """CLI entry point: generate, save and summarise the corpus."""
    parser = argparse.ArgumentParser(
        description="Generate the MailMind synthetic email dataset."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=config.SAMPLES_PER_CATEGORY,
        help="rows per category (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(config.DATASET_PATH),
        help="output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--ambiguity",
        type=float,
        default=config.DATASET_AMBIGUITY,
        help="fraction of hard/overlapping rows (default: %(default)s)",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=config.DATASET_LABEL_NOISE,
        help="fraction of label-noise rows (default: %(default)s)",
    )
    args = parser.parse_args()

    df = generate_dataset(
        samples_per_category=args.samples,
        seed=args.seed,
        ambiguity=args.ambiguity,
        label_noise=args.label_noise,
    )
    save_dataset(df, args.out)
    print(dataset_summary(df))
    print(f"\nSaved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
