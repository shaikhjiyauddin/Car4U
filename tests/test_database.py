"""
Tests for src/database.py (Marketplace: users, listings, messages)

Run with: pytest tests/ -v
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import database as db


@pytest.fixture
def test_db(tmp_path):
    """A fresh, isolated SQLite DB file for each test."""
    path = str(tmp_path / "test_car4u.db")
    db.init_db(path)
    return path


@pytest.fixture
def seller_and_buyer(test_db):
    db.create_user('Wajahat', 'wajahat@example.com', 'password123', test_db)
    db.create_user('Riya', 'riya@example.com', 'password456', test_db)
    seller = db.verify_login('wajahat@example.com', 'password123', test_db)
    buyer = db.verify_login('riya@example.com', 'password456', test_db)
    return seller, buyer, test_db


class TestUsers:
    def test_create_user_success(self, test_db):
        ok, msg = db.create_user('Alice', 'alice@example.com', 'securepass', test_db)
        assert ok is True

    def test_duplicate_email_rejected(self, test_db):
        db.create_user('Alice', 'alice@example.com', 'securepass', test_db)
        ok, msg = db.create_user('Someone Else', 'alice@example.com', 'otherpass', test_db)
        assert ok is False
        assert 'already exists' in msg.lower()

    def test_short_password_rejected(self, test_db):
        ok, msg = db.create_user('Bob', 'bob@example.com', '123', test_db)
        assert ok is False

    def test_missing_fields_rejected(self, test_db):
        ok, msg = db.create_user('', 'x@example.com', 'password1', test_db)
        assert ok is False

    def test_login_correct_credentials(self, test_db):
        db.create_user('Alice', 'alice@example.com', 'securepass', test_db)
        user = db.verify_login('alice@example.com', 'securepass', test_db)
        assert user is not None
        assert user['name'] == 'Alice'

    def test_login_wrong_password(self, test_db):
        db.create_user('Alice', 'alice@example.com', 'securepass', test_db)
        user = db.verify_login('alice@example.com', 'wrongpass', test_db)
        assert user is None

    def test_login_nonexistent_email(self, test_db):
        user = db.verify_login('nobody@example.com', 'whatever1', test_db)
        assert user is None

    def test_login_case_insensitive_email(self, test_db):
        db.create_user('Alice', 'alice@example.com', 'securepass', test_db)
        user = db.verify_login('ALICE@EXAMPLE.COM', 'securepass', test_db)
        assert user is not None


class TestListings:
    def test_add_and_get_listing(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        listing_id = db.add_listing(
            seller['id'], 'Maruti', 'Swift', 2018, 4.5, 'Petrol', 'Manual',
            35000, 'Ahmedabad', 'Well maintained.', test_db
        )
        listing = db.get_listing(listing_id, test_db)
        assert listing['brand'] == 'Maruti'
        assert listing['owner_name'] == 'Wajahat'
        assert listing['photo'] is None

    def test_get_all_listings_returns_newest_first(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        id1 = db.add_listing(seller['id'], 'Honda', 'City', 2015, 5.0, 'Petrol', 'Manual',
                              50000, 'Mumbai', '', test_db)
        id2 = db.add_listing(seller['id'], 'Tata', 'Nexon', 2021, 9.0, 'Diesel', 'Automatic',
                              10000, 'Pune', '', test_db)
        listings = db.get_all_listings(test_db)
        assert listings[0]['id'] == id2  # most recent first

    def test_get_listings_by_owner(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        db.add_listing(seller['id'], 'Honda', 'City', 2015, 5.0, 'Petrol', 'Manual',
                        50000, 'Mumbai', '', test_db)
        db.add_listing(buyer['id'], 'Tata', 'Nexon', 2021, 9.0, 'Diesel', 'Automatic',
                        10000, 'Pune', '', test_db)
        seller_listings = db.get_listings_by_owner(seller['id'], test_db)
        assert len(seller_listings) == 1
        assert seller_listings[0]['brand'] == 'Honda'

class TestMessages:
    def test_send_and_retrieve_conversation(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        listing_id = db.add_listing(seller['id'], 'Maruti', 'Swift', 2018, 4.5, 'Petrol',
                                     'Manual', 35000, 'Ahmedabad', '', test_db)
        db.send_message(listing_id, buyer['id'], seller['id'], 'Is this available?', test_db)
        db.send_message(listing_id, seller['id'], buyer['id'], 'Yes!', test_db)

        convo = db.get_conversation(listing_id, buyer['id'], seller['id'], test_db)
        assert len(convo) == 2
        assert convo[0]['body'] == 'Is this available?'
        assert convo[1]['body'] == 'Yes!'

    def test_empty_message_rejected(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        listing_id = db.add_listing(seller['id'], 'Maruti', 'Swift', 2018, 4.5, 'Petrol',
                                     'Manual', 35000, 'Ahmedabad', '', test_db)
        sent = db.send_message(listing_id, buyer['id'], seller['id'], '   ', test_db)
        assert sent is False

    def test_conversation_isolated_per_listing(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        listing1 = db.add_listing(seller['id'], 'Maruti', 'Swift', 2018, 4.5, 'Petrol',
                                   'Manual', 35000, 'Ahmedabad', '', test_db)
        listing2 = db.add_listing(seller['id'], 'Honda', 'City', 2015, 5.0, 'Petrol',
                                   'Manual', 50000, 'Mumbai', '', test_db)
        db.send_message(listing1, buyer['id'], seller['id'], 'About the Swift', test_db)
        db.send_message(listing2, buyer['id'], seller['id'], 'About the City', test_db)

        convo1 = db.get_conversation(listing1, buyer['id'], seller['id'], test_db)
        convo2 = db.get_conversation(listing2, buyer['id'], seller['id'], test_db)
        assert len(convo1) == 1 and convo1[0]['body'] == 'About the Swift'
        assert len(convo2) == 1 and convo2[0]['body'] == 'About the City'

    def test_user_conversations_appear_for_both_sides(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        listing_id = db.add_listing(seller['id'], 'Maruti', 'Swift', 2018, 4.5, 'Petrol',
                                     'Manual', 35000, 'Ahmedabad', '', test_db)
        db.send_message(listing_id, buyer['id'], seller['id'], 'Hi', test_db)

        buyer_inbox = db.get_user_conversations(buyer['id'], test_db)
        seller_inbox = db.get_user_conversations(seller['id'], test_db)
        assert len(buyer_inbox) == 1
        assert len(seller_inbox) == 1
        assert buyer_inbox[0]['other_user_name'] == 'Wajahat'
        assert seller_inbox[0]['other_user_name'] == 'Riya'

    def test_no_conversations_returns_empty_list(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        assert db.get_user_conversations(buyer['id'], test_db) == []


class TestPipelineStats:
    def test_empty_db_returns_zeroes(self, test_db):
        stats = db.get_pipeline_stats(test_db)
        assert stats['total_users'] == 0
        assert stats['total_listings'] == 0
        assert stats['total_messages'] == 0
        assert stats['exported_to_training'] == 0
        assert stats['pending_export'] == 0
        assert stats['latest_listing_at'] is None

    def test_stats_reflect_activity(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        l1 = db.add_listing(seller['id'], 'Maruti', 'Swift', 2019, 5.2, 'Petrol', 'Manual',
                             25000, 'Ahmedabad', '', test_db)
        db.add_listing(seller['id'], 'Honda', 'City', 2020, 8.0, 'Diesel', 'Automatic',
                        15000, 'Mumbai', '', test_db)
        db.send_message(l1, buyer['id'], seller['id'], 'Hi!', test_db)

        stats = db.get_pipeline_stats(test_db)
        assert stats['total_users'] == 2
        assert stats['total_listings'] == 2
        assert stats['total_messages'] == 1
        assert stats['pending_export'] == 2

    def test_export_updates_pending_count(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        l1 = db.add_listing(seller['id'], 'Maruti', 'Swift', 2019, 5.2, 'Petrol', 'Manual',
                             25000, 'Ahmedabad', '', test_db)
        db.mark_listings_exported([l1], test_db)

        stats = db.get_pipeline_stats(test_db)
        assert stats['exported_to_training'] == 1
        assert stats['pending_export'] == 0

    def test_listings_timeline_ordered_oldest_first(self, seller_and_buyer):
        seller, buyer, test_db = seller_and_buyer
        db.add_listing(seller['id'], 'Maruti', 'Swift', 2019, 5.2, 'Petrol', 'Manual',
                        25000, 'Ahmedabad', '', test_db)
        db.add_listing(seller['id'], 'Honda', 'City', 2020, 8.0, 'Diesel', 'Automatic',
                        15000, 'Mumbai', '', test_db)
        timeline = db.get_listings_timeline(test_db)
        assert len(timeline) == 2
        assert timeline[0]['brand'] == 'Maruti'  # created first
        assert timeline[1]['brand'] == 'Honda'
