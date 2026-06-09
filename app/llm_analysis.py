
# from openai import OpenAI

# client = OpenAI(api_key="YOUR_API_KEY")

# def analyze_therapist_response(patient_text, therapist_response):

#     prompt = f'''
#     Analyze therapist response quality.

#     Patient:
#     {patient_text}

#     Therapist:
#     {therapist_response}
#     '''

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     return response.choices[0].message.content
# Free therapist response analysis
# No OpenAI API required

def analyze_therapist_response(patient_text, therapist_response):

    score = 0
    feedback = []

    # Simple empathy keyword checks
    empathy_words = [
        "समझ",
        "support",
        "help",
        "care",
        "sorry",
        "feel",
        "understand"
    ]

    response_lower = therapist_response.lower()

    for word in empathy_words:
        if word.lower() in response_lower:
            score += 1

    # Generate feedback
    if score >= 5:
        quality = "Excellent"
    elif score >= 3:
        quality = "Good"
    else:
        quality = "Needs Improvement"

    result = f'''
    Therapist Response Quality Analysis

    Quality: {quality}

    Empathy Score: {score}/7

    Feedback:
    - Response analyzed successfully
    - Therapist empathy evaluated
    - Emotional support checked
    '''

    return result
