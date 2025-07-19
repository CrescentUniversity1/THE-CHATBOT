import re

def detect_intent(text):
    text = text.lower()

    # Small talk
    if any(phrase in text for phrase in ["hello", "hi", "good morning", "good afternoon", "thanks", "thank you", "how are you"]):
        return "small_talk"

    # Academic queries
    if any(word in text for word in ["course", "courses", "level", "department", "faculty", "semester"]):
        return "academic"

    # Admin / support queries
    if any(word in text for word in ["register", "registration", "fees", "payment", "result", "exam", "hostel", "portal"]):
        return "admin"

    # Unknown or fallback
    return "unknown"
