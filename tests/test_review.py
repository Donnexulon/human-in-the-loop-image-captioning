import sqlite3

from caption_review.review import initialize_database, save_review


def test_review_is_saved_to_sqlite(tmp_path) -> None:
    database = tmp_path / "reviews.sqlite"
    initialize_database(database)
    save_review(database, "sample-1", tmp_path / "image.png", "a test image", 4, False, "", "Osama")
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT image_id, rating, hallucination, reviewer FROM reviews").fetchone()
    assert row == ("sample-1", 4, 0, "Osama")
