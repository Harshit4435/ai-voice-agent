"""Unit Tests for Entity Extraction Engine."""
import unittest
from app.ml.entity_extractor import InboundEntityExtractor


class TestEntityExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = InboundEntityExtractor()

    def test_extract_name_patterns(self):
        # Pattern: my name is
        t1 = "Hello, my name is John Miller and I'm calling about the contract."
        self.assertEqual(self.extractor.extract_name(t1), "John Miller")

        # Pattern: this is
        t2 = "Good morning, this is Dr. Sarah Jenkins calling to discuss the clinical trial."
        self.assertIn("Sarah Jenkins", self.extractor.extract_name(t2))

        # Pattern: [Name] here
        t3 = "Elena Rostova here, calling regarding the partnership."
        self.assertEqual(self.extractor.extract_name(t3), "Elena Rostova")

    def test_extract_reason(self):
        t1 = "Hi, my name is John Miller and I'm calling regarding the commercial contract proposal."
        reason = self.extractor.extract_reason(t1)
        self.assertIsNotNone(reason)
        self.assertIn("commercial contract proposal", reason)

        t2 = "Calling to discuss the enterprise software demo we scheduled."
        reason2 = self.extractor.extract_reason(t2)
        self.assertIsNotNone(reason2)
        self.assertIn("enterprise software demo", reason2)

    def test_extract_phone_number(self):
        t1 = "My number is 415-555-0199, reach back whenever."
        self.assertEqual(self.extractor.extract_phone(t1), "415-555-0199")

        t2 = "Call me at +1 (650) 555-7821 tomorrow."
        self.assertEqual(self.extractor.extract_phone(t2), "+1 (650) 555-7821")

        t3 = "No phone number here at all."
        self.assertIsNone(self.extractor.extract_phone(t3))

        # Space-separated 4-4-2 format
        t4 = "the best time would be tomorrow at 2 and my number is 5550 1928 34"
        phone4 = self.extractor.extract_phone(t4)
        self.assertIsNotNone(phone4)
        self.assertEqual("".join(c for c in phone4 if c.isdigit()), "5550192834")

        # 2-2-2-2-2 format
        t5 = "the phone number is 55 50 19 28 34"
        phone5 = self.extractor.extract_phone(t5)
        self.assertIsNotNone(phone5)
        self.assertEqual("".join(c for c in phone5 if c.isdigit()), "5550192834")

        # Spoken words
        t6 = "my number is five five five zero one nine two eight three four"
        phone6 = self.extractor.extract_phone(t6)
        self.assertIsNotNone(phone6)
        self.assertEqual("".join(c for c in phone6 if c.isdigit()), "5550192834")

    def test_extract_best_time(self):
        t1 = "You can reach me at 415-555-0199, and the best time to call is tomorrow at 2 PM."
        best_time = self.extractor.extract_best_time(t1)
        self.assertIsNotNone(best_time)
        self.assertIn("tomorrow at 2 PM", best_time)

        t2 = "I am available tomorrow morning."
        best_time2 = self.extractor.extract_best_time(t2)
        self.assertIsNotNone(best_time2)
        self.assertIn("tomorrow morning", best_time2)

    def test_detect_urgency(self):
        t1 = "Normal general check-in call."
        self.assertEqual(self.extractor.detect_urgency(t1), "NORMAL")

        t2 = "Emergency! Our main production server cluster has crashed!"
        self.assertEqual(self.extractor.detect_urgency(t2), "URGENT")

    def test_extract_all_combined(self):
        text = "Hi, my name is Michael Chang. I'm calling regarding the cloud migration. My number is 510-555-9012 and best time to call is tomorrow afternoon."
        entities = self.extractor.extract_all(text)
        self.assertEqual(entities.caller_name, "Michael Chang")
        self.assertIn("cloud migration", entities.reason_for_calling)
        self.assertEqual(entities.phone_number, "510-555-9012")
        self.assertIn("tomorrow afternoon", entities.best_time_to_reach)
        self.assertEqual(entities.urgency_level, "NORMAL")


if __name__ == "__main__":
    unittest.main()

