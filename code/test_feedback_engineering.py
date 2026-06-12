"""
DeepEval Test Suite for Feedback-to-Engineering-Insights Pipeline
==================================================================
Run with: python test_feedback_engineering.py

Prerequisites:
  pip install deepeval openai
  Set OPENAI_API_KEY as environment variable (do NOT hardcode)
    Windows PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."
    Mac/Linux:           export OPENAI_API_KEY="sk-proj-..."
"""
import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

os.environ["OPENAI_API_KEY"] = "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"
if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Set OPENAI_API_KEY environment variable first."); sys.exit(1)

TEST_FEEDBACK = {
    "F1_bug_checkout": {"input": "The checkout page keeps freezing when I try to apply a discount code on my iPhone 14. Happens every time. I've lost 3 orders this week because of it. Using Safari, latest iOS. -- App Store review, 1 star", "label": "detailed"},
    "F2_perf_api": {"input": "The API response times have degraded significantly since last month. Our p95 latency went from 200ms to 800ms. We're on the Enterprise plan and this is affecting our production systems. We've opened 3 tickets about this already. Account: api-team@bigcorp.com -- Support ticket", "label": "detailed"},
    "F3_usability_onboarding": {"input": "The onboarding flow is confusing. I signed up yesterday and still can't figure out how to create my first project. The getting started guide links to a 404 page. I watched 3 YouTube tutorials and they all show a different UI than what I see. Free plan, Chrome on Windows. -- Support ticket", "label": "detailed"},
    "F4_security": {"input": "Login alert at 3:47 AM EST from IP 203.0.113.42 (Romania). I did NOT log in. Account: lisa.t@company.com, #AC-41209. Changed password, enabled 2FA. Need: 1) Was data accessed? 2) 30-day audit log 3) Other accounts affected? -- Support ticket", "label": "detailed"},
    "F5_vague_angry": {"input": "This app is garbage. Total waste of money.", "label": "vague"},
    "F6_vague_minimal": {"input": "Terrible. Just terrible. 0 stars if I could.", "label": "vague"},
    "F7_vague_indifferent": {"input": "Meh. It's fine I guess.", "label": "vague"},
    "F8_vague_demand": {"input": "Fix your stuff.", "label": "vague"},
}

# ============================================================
# PASTE YOUR N8N PIPELINE OUTPUTS HERE
# ============================================================
# PIPELINE_OUTPUTS = {
#     "F1_bug_checkout": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F2_perf_api": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F3_usability_onboarding": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F4_security": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F5_vague_angry": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F6_vague_minimal": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F7_vague_indifferent": """PASTE FULL PIPELINE OUTPUT HERE""",
#     "F8_vague_demand": """PASTE FULL PIPELINE OUTPUT HERE""",
# }

# sample outputs for testing the evaluation framework using F1-F8 test use-cases where output is collected using basemodel (before fine-tuning).
PIPELINE_OUTPUTS = {
    "F1_bug_checkout": """=== ENGINEERING INSIGHT ===\nTITLE: Checkout freeze on iOS Safari when applying discount codes\nTICKET_TYPE: BUG\nSEVERITY: HIGH\n\nTECHNICAL SUMMARY: The checkout page freezes when attempting to apply a discount code on an iPhone 14 running the latest iOS version, specifically using Safari. This issue occurs consistently and has resulted in lost orders. The exact cause of the freeze is unknown.\n\nAFFECTED COMPONENTS:\n- System: Checkout\n- Platform: iOS\n- Environment: Mobile\n\nREPRODUCTION STEPS:\n1. Open the app on an iPhone 14 with the latest iOS version.\n2. Navigate to the checkout page.\n3. Attempt to apply a discount code using Safari.\n\nINVESTIGATION CHECKLIST:\n- [ ] Review Safari-specific JavaScript interactions and potential conflicts.\n- [ ] Investigate checkout API calls and server-side processing for any errors or timeouts.\n- [ ] Check for any iOS-specific security restrictions or limitations that may be causing the freeze.\n\nDATA GAPS FOR ENGINEERING:\n- More detailed logs or error messages from the iOS device to understand the cause of the freeze.\n- Information on the specific discount codes used and any variations that may affect the issue.\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: This issue has resulted in lost orders and has a high blast radius, impacting multiple users on a critical path of the application.""",
    "F2_perf_api": """=== ENGINEERING INSIGHT ===\nTITLE: Increased API response times on Enterprise plan\nTICKET_TYPE: PERFORMANCE\nSEVERITY: P2-HIGH\nTECHNICAL SUMMARY: API response times have degraded from 200ms to 800ms, affecting production systems on the Enterprise plan. This change is likely due to an increase in latency, impacting the overall system performance. Further investigation is required to determine the root cause.\n\nAFFECTED COMPONENTS:\n- System: API\n- Platform: Enterprise plan\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. Check API response times for the past month to confirm the degradation.\n2. Verify if the issue is specific to the Enterprise plan or affects other plans as well.\n\nINVESTIGATION CHECKLIST:\n- Review API logs for any changes or anomalies in the past month\n- Check for any recent code deployments or changes to the API\n- Investigate resource utilization and CPU/Memory usage on the API servers\n\nDATA GAPS FOR ENGINEERING:\n- Detailed analysis of API request patterns and volumes\n- Information on recent infrastructure changes or upgrades\n- Any relevant metrics or logs from the past month\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The degradation in API response times has a significant impact on production systems, affecting the overall performance and potentially causing user experience issues. Given the Enterprise plan is affected, this issue requires immediate attention to ensure minimal disruption to critical systems.""",
    "F3_usability_onboarding": """=== ENGINEERING INSIGHT ===\nTITLE: Confusing onboarding flow and broken getting started guide\nTICKET_TYPE: BUG\nSEVERITY: MEDIUM\nTECHNICAL SUMMARY: The onboarding flow is experiencing issues, causing difficulty in creating a first project. The getting started guide links to a 404 page, and YouTube tutorials are showing a different UI than the current application. This discrepancy is affecting users on the free plan, running Chrome on Windows.\n\nAFFECTED COMPONENTS:\n- System: Onboarding flow\n- Platform: Chrome\n- Environment: Windows\n\nREPRODUCTION STEPS:\n1. Signed up for the service yesterday\n2. Attempted to follow the getting started guide\n3. Encountered a 404 page\n4. Watched three YouTube tutorials that showed a different UI\n5. Still unable to create a first project\n\nINVESTIGATION CHECKLIST:\n- [ ] Verify the getting started guide URL is correct\n- [ ] Check for any recent changes to the onboarding flow or UI\n- [ ] Compare the current UI with the YouTube tutorials to identify discrepancies\n- [ ] Investigate why the YouTube tutorials are showing a different UI\n\nDATA GAPS FOR ENGINEERING:\n- More information about the exact steps taken during onboarding\n- Details about the exact YouTube tutorials watched (e.g., video IDs, timestamps)\n- Any error messages or logs related to the 404 page\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The onboarding flow is critical for new users, and a confusing experience can lead to user frustration and potential churn.""",
    "F4_security": """=== ENGINEERING INSIGHT ===\nTITLE: Unauthorized login attempt from Romania\nTICKET_TYPE: SECURITY\nSEVERITY: HIGH\nTECHNICAL SUMMARY: An unauthorized login attempt was made to an account (lisa.t@company.com, #AC-41209) from an unknown device in Romania, despite the account having a changed password and enabled 2FA. The incident occurred at 3:47 AM EST. We need to determine if any data was accessed and perform a 30-day audit log review.\n\nAFFECTED COMPONENTS:\n- System: Authentication\n- Platform: Web\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. An unknown individual attempted to log in to the account lisa.t@company.com from IP address 203.0.113.42 in Romania at 3:47 AM EST.\n2. The login attempt was made despite the account having a changed password and enabled 2FA.\n\nINVESTIGATION CHECKLIST:\n- Check the authentication logs for any successful login attempts or data access\n- Review the 30-day audit log for the account #AC-41209\n- Investigate if any other accounts were accessed or affected by the same IP address\n\nDATA GAPS FOR ENGINEERING:\n- More information about the device used for the login attempt\n- Details about the login attempt, such as the user agent and browser type\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The unauthorized login attempt has high severity due to the potential data exposure and security breach. We need to act quickly to determine the extent of the incident and prevent further unauthorized access.""",
    "F5_vague_angry": """== ENGINEERING INSIGHT ===\nTITLE: Unclear technical issue with app functionality\nTICKET_TYPE: INVESTIGATION_NEEDED\nSEVERITY: UNKNOWN\nTECHNICAL SUMMARY: Insufficient feedback for engineering action. User reports the app is "garbage" and a "waste of money", but no specific technical details are provided.\n\nAFFECTED COMPONENTS:\n- System: Entire application\n- Platform: Not specified\n- Environment: Not specified\n\nREPRODUCTION STEPS:\n1. User reports the app is not functioning as expected, but no specific steps are provided.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Request a clear description of the issue and any error messages encountered\n\nDATA GAPS FOR ENGINEERING:\n- Specific technical details about the issue, including any error messages, stack traces, or system logs\n- Information about the user's environment, including platform and device\n\nPRIORITY RECOMMENDATION: P4-LOW\nPRIORITY JUSTIFICATION: The issue is unclear and no specific impact on users or data integrity is reported, but further investigation is needed to determine the nature of the issue.""",
    "F6_vague_minimal": """=== ENGINEERING INSIGHT ===\nTITLE: Poor user experience with checkout process\nTICKET_TYPE: BUG\nSEVERITY: HIGH\nTECHNICAL SUMMARY: The user experienced a poor checkout experience, specifically stating that the process was 'terrible'. The exact nature of the issue is unclear, but it occurred during the checkout process. The user did not provide specific details about the issue.\n\nAFFECTED COMPONENTS:\n- System: Checkout\n- Platform: Web\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. The user attempted to check out on the website.\n2. The user reported a poor experience, but did not provide further details.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Review recent checkout process changes\n\nDATA GAPS FOR ENGINEERING:\n- Specific error messages or error codes encountered by the user\n- Steps taken by the user leading up to the issue\n- Any relevant system logs or error messages\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The user reported a high-severity issue with the checkout process, which is a critical part of the user experience. Further investigation is needed to determine the root cause and prevent similar issues in the future.""",
    "F7_vague_indifferent": """=== ENGINEERING INSIGHT ===\nTITLE: User reports issue with the app experience\nTICKET_TYPE: INVESTIGATION_NEEDED\nSEVERITY: Unknown - single report\nTECHNICAL SUMMARY: Insufficient feedback for engineering action. Further details are required to determine the nature of the issue.\n\nAFFECTED COMPONENTS:\n- System: Unknown\n- Platform: Unknown\n- Environment: Unknown\n\nREPRODUCTION STEPS:\n1. User reported the issue with the app experience, but no specific steps were provided.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Gather information about the user's environment, platform, and system\n\nDATA GAPS FOR ENGINEERING:\n- Specific details about the issue\n- Technical information about the user's environment, platform, and system\n\nPRIORITY RECOMMENDATION: P4-LOW\nPRIORITY JUSTIFICATION: Insufficient information is available to determine the priority of this issue. Further investigation is required to understand the scope and impact.""",
    "F8_vague_demand": """=== ENGINEERING INSIGHT ===\nTITLE: iOS App Crashes When Applying Discount Codes\nTICKET_TYPE: BUG\nSEVERITY: HIGH\n\nTECHNICAL SUMMARY: The iOS app crashes when users attempt to apply discount codes on the checkout page. The crash occurs on the latest version of iOS and Safari. Further investigation is required to determine the root cause.\n\nAFFECTED COMPONENTS:\n- System: Checkout\n- Platform: iOS\n- Environment: Safari\n\nREPRODUCTION STEPS:\n1. Users report that they were able to select a product and proceed to the checkout page.\n2. When attempting to apply a discount code on the checkout page, the app crashes.\n\nINVESTIGATION CHECKLIST:\n- Check the app's crash logs for any relevant error messages or stack traces.\n- Review the code for the checkout page to ensure it is correctly handling discount code input.\n- Verify that the discount code API is functioning correctly.\n\nDATA GAPS FOR ENGINEERING:\n- Crash logs from the affected users\n- Additional details on the specific discount code being used when the crash occurred\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The crash is preventing users from completing the checkout process, which is a critical path for revenue generation. The high severity of the issue warrants immediate attention to minimize the impact on users and business operations.""",
}

# sample outputs for testing the evaluation framework using F1-F8 test use-cases where output is collected using fine tuned models.
# PIPELINE_OUTPUTS = {
#     "F1_bug_checkout": """=== ENGINEERING INSIGHT ===\nTITLE: Checkout freeze on iOS Safari when applying discount codes\nTICKET_TYPE: BUG\nSEVERITY: CRITICAL\nTECHNICAL SUMMARY: The checkout page freezes on iOS Safari when attempting to apply a discount code, resulting in lost orders. This issue occurs on the latest iOS version. The freeze is consistent, indicating a potential rendering or JavaScript issue.\n\nAFFECTED COMPONENTS:\n- System: E-commerce checkout system\n- Platform: iOS\n- Environment: Mobile Safari\n\nREPRODUCTION STEPS:\n1. Launch the app on iOS using Safari\n2. Navigate to the checkout page\n3. Attempt to apply a discount code\n\nINVESTIGATION CHECKLIST:\n- [ ] Review JavaScript code handling discount code application\n- [ ] Inspect Safari's console for error messages\n- [ ] Verify rendering performance on the checkout page\n\nDATA GAPS FOR ENGINEERING:\n- Detailed logs or crash reports from the affected iOS devices\n- Information about the specific discount codes causing the issue\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The issue results in lost orders, impacting customer satisfaction and potentially revenue, making it a critical priority to resolve.""",
#     "F2_perf_api": """=== ENGINEERING INSIGHT ===\nTITLE: API Response Time Degradation on Enterprise Plan\nTICKET_TYPE: PERFORMANCE\nSEVERITY: CRITICAL\nTECHNICAL SUMMARY: API response times have increased significantly from 200ms to 800ms, affecting production systems on the Enterprise plan. This degradation is impacting the overall system performance and requires immediate investigation. The root cause is unclear.\n\nAFFECTED COMPONENTS:\n- System: API\n- Platform: Enterprise Plan\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. The issue is observed in the production environment on the Enterprise plan.\n2. The degradation is noticeable since last month.\n3. Three support tickets have been opened about this issue.\n\nINVESTIGATION CHECKLIST:\n- Review system logs for any changes or anomalies around the time of degradation\n- Check for any recent updates or deployments that could be contributing to the issue\n- Verify if the Enterprise plan is experiencing any unique configuration or resource constraints\n\nDATA GAPS FOR ENGINEERING:\n- Specific metrics or data points that led to the conclusion of a 600ms increase in p95 latency\n- Any recent changes or updates made to the API or underlying infrastructure\n- Information on the current resource utilization and configuration of the Enterprise plan\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The significant increase in API response times is impacting the overall system performance and requires immediate attention to prevent further degradation and potential service disruptions.""",
#     "F3_usability_onboarding": """=== ENGINEERING INSIGHT ===\nTITLE: Confusing onboarding flow on Free plan, Chrome on Windows\nTICKET_TYPE: FEATURE\nSEVERITY: MEDIUM\nTECHNICAL SUMMARY: The onboarding flow appears to be inconsistent, with a 404 page being displayed when attempting to access the getting started guide. This is affecting users on the Free plan, running Chrome on Windows, who are unable to create their first project. The UI discrepancy between YouTube tutorials and the actual application is causing confusion.\n\nAFFECTED COMPONENTS:\n- System: Onboarding flow\n- Platform: Web (Chrome on Windows)\n- Environment: Free plan\n\nREPRODUCTION STEPS:\n1. Signed up for a new account on the Free plan\n2. Attempted to access the getting started guide\n3. Encountered a 404 page\n4. Watched 3 YouTube tutorials, which displayed a different UI than the actual application\n\nINVESTIGATION CHECKLIST:\n- Verify the getting started guide URL and ensure it is not a 404 page\n- Compare the UI in the YouTube tutorials with the current application UI to identify any discrepancies\n- Review the onboarding flow for Free plan users to identify any inconsistencies\n\nDATA GAPS FOR ENGINEERING:\n- Specific details about the discrepancies between the YouTube tutorials and the actual application UI\n- Information about the getting started guide URL and its intended functionality\n\nPRIORITY RECOMMENDATION: P2-HIGH\nPRIORITY JUSTIFICATION: The onboarding flow is critical for new users, and a confusing experience can lead to frustration and a higher likelihood of churn. The discrepancy in UI between tutorials and the application can also lead to user confusion and decreased trust in the application.""",
#     "F4_security": """=== ENGINEERING INSIGHT ===\nTITLE: Unsuccessful login attempt from unknown IP with compromised account\nTICKET_TYPE: SECURITY INCIDENT\nSEVERITY: HIGH\n\nTECHNICAL SUMMARY: There was an unsuccessful login attempt from an unknown IP address (203.0.113.42) from Romania at 3:47 AM EST for account lisa.t@company.com (account #AC-41209), which has a changed password and enabled 2FA. The login attempt may have been unauthorized, and we need to determine if data was accessed.\n\nAFFECTED COMPONENTS:\n- System: Authentication\n- Platform: Web\n- Environment: Production\n\nREPRODUCTION STEPS:\n1. Attempted login to account lisa.t@company.com from IP 203.0.113.42\n2. Login attempt was unsuccessful due to changed password and 2FA\n\nINVESTIGATION CHECKLIST:\n- Review authentication logs for the IP address 203.0.113.42\n- Check if there are any successful login attempts from this IP address for other accounts\n- Verify if the account lisa.t@company.com was accessed or modified\n\nDATA GAPS FOR ENGINEERING:\n- Details of the login attempt (e.g., login time, browser type, device information)\n- Any relevant logs or system activity related to the IP address 203.0.113.42\n\nPRIORITY RECOMMENDATION: P1-CRITICAL\nPRIORITY JUSTIFICATION: The incident may involve unauthorized access to sensitive user data, which poses a high risk to user security and confidentiality.""",
#     "F5_vague_angry": """=== ENGINEERING INSIGHT ===\nTITLE: User Experience Issue - App Perception\nTICKET_TYPE: INVESTIGATION_NEEDED\nSEVERITY: UNCLASSIFIED\n\nTECHNICAL SUMMARY: Insufficient feedback for engineering action. The user perceives the app as garbage and a waste of money, but no specific technical issues are mentioned.\n\nAFFECTED COMPONENTS:\n- System: Mobile Application\n- Platform: Unknown\n- Environment: Unknown\n\nREPRODUCTION STEPS:\n1. User used the app\n2. User experienced issues with the app\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Gather information on specific features or functionalities that led to this perception\n\nDATA GAPS FOR ENGINEERING:\n- Specific technical issues or features causing user dissatisfaction\n- Information about the user's device, platform, and environment\n\nPRIORITY RECOMMENDATION: P4-LOW\nPRIORITY JUSTIFICATION: This issue is low priority as it is a subjective user experience issue without any specific technical details, and it's unclear if it affects a significant number of users.""",
#     "F6_vague_minimal": """=== ENGINEERING INSIGHT ===\nTITLE: Poor user experience with our platform\nTICKET_TYPE: BUG\nSEVERITY: UNCLASSIFIED\nTECHNICAL SUMMARY: The user experienced a poor user experience with our platform, citing 'terrible' as the reason. The exact nature of the issue is unclear. No specific technical details were provided.\n\nAFFECTED COMPONENTS:\n- System: Unknown\n- Platform: Unknown\n- Environment: Unknown\n\nREPRODUCTION STEPS:\n1. The user used our platform and experienced a poor experience.\n2. They described the experience as 'terrible'.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details\n- Gather more information about the user's actions and environment\n\nDATA GAPS FOR ENGINEERING:\n- Technical details about the issue\n- Specific actions taken by the user\n- Platform, system, and environment information\n\nPRIORITY RECOMMENDATION: P3-MEDIUM\nPRIORITY JUSTIFICATION: Insufficient feedback for engineering action.""",
#     "F7_vague_indifferent": """=== ENGINEERING INSIGHT ===\nTITLE: General dissatisfaction with the application's performance\nTICKET_TYPE: INVESTIGATION_NEEDED\nSEVERITY: UNKNOWN\nTECHNICAL SUMMARY: Insufficient feedback for engineering action. The user expressed general dissatisfaction but did not provide specific details about the issue. Further clarification is required to determine the nature of the problem.\n\nAFFECTED COMPONENTS:\n- System: Unknown\n- Platform: Unknown\n- Environment: Unknown\n\nREPRODUCTION STEPS:\n1. The user mentioned that the application is 'fine, I guess'.\n\nINVESTIGATION CHECKLIST:\n- Reach out to user for technical details about the issue\n- Clarify what specifically is not meeting expectations\n\nDATA GAPS FOR ENGINEERING:\n- Specific symptoms or error messages encountered\n- Steps to reproduce the issue\n- Technical details about the environment (e.g., device, OS, browser)\n\nPRIORITY RECOMMENDATION: P4-LOW\nPRIORITY JUSTIFICATION: The user's feedback is too vague to determine the impact on users or the severity of the issue. Further clarification is needed before a more informed priority can be assigned.""",
#     "F8_vague_demand": """I need more information about the issue you're experiencing. Can you please provide a detailed description of the problem, including the steps you took to encounter it? This will help me write a more accurate and helpful engineering insight.\n\nPlease provide the following:\n\n- What is the problem you're experiencing?\n- What steps did you take to encounter the issue?\n- What platform and environment are you using (e.g., iOS, Android, web, Safari, Chrome, etc.)?\n\nOnce I have this information, I can help you write a structured engineering insight.""",
# }

# ============================================================
# METRICS
# ============================================================
engineering_groundedness = GEval(
    name="Engineering Groundedness",
    criteria=(
        "The input is detailed user feedback with specific technical details. "
        "Does the engineering insight ONLY use information from this feedback? "
        "1) REPRODUCTION STEPS must only describe actions the user actually mentioned. "
        "2) No invented internal system names or architecture details the user did not mention "
        "(e.g., do not invent 'API Gateway' or 'GraphQL' if the user just said 'API'). "
        "3) No fabricated user counts, percentages, or revenue impact. "
        "4) DATA GAPS section should list what is NOT known. "
        "Grounded = PASS. Invented technical specifics = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

engineering_usefulness = GEval(
    name="Engineering Usefulness",
    criteria=(
        "Would an engineer find this insight useful for starting an investigation? "
        "Does it identify the affected system based on what the user described? "
        "Does it provide a reasonable investigation checklist? "
        "Does it clearly state data gaps? "
        "Useful engineering ticket = PASS. Vague or unhelpful = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

vague_handling = GEval(
    name="Vague Feedback Handling",
    criteria=(
        "The input is vague or lacks technical details. A CORRECT response should: "
        "1) Identify that engineering cannot act without more information. "
        "2) Set affected components to UNKNOWN. "
        "3) NOT invent systems, platforms, error types, or root causes. "
        "4) NOT fabricate reproduction steps. "
        "5) Recommend reaching out to the user for clarification. "
        "6) Set priority to LOW since nothing is actionable. "
        "The checklist CAN include multiple follow-up items about GATHERING info "
        "(ask for platform, request screenshots) as long as no specific problem is invented. "
        "Correctly flagging as unactionable = PASS. Inventing details = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4o",
)

relevancy = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")

def run_evaluation():
    print("\n" + "=" * 60)
    print("EVALUATING DETAILED FEEDBACK (4 test cases)")
    print("=" * 60)
    detailed = [LLMTestCase(input=d["input"], actual_output=PIPELINE_OUTPUTS[f])
                for f, d in TEST_FEEDBACK.items()
                if d["label"] == "detailed" and "PASTE" not in PIPELINE_OUTPUTS.get(f, "PASTE")]
    if detailed:
        evaluate(test_cases=detailed, metrics=[engineering_groundedness, engineering_usefulness, relevancy])

    print("\n" + "=" * 60)
    print("EVALUATING VAGUE FEEDBACK (4 test cases)")
    print("=" * 60)
    vague = [LLMTestCase(input=d["input"], actual_output=PIPELINE_OUTPUTS[f])
             for f, d in TEST_FEEDBACK.items()
             if d["label"] == "vague" and "PASTE" not in PIPELINE_OUTPUTS.get(f, "PASTE")]
    if vague:
        evaluate(test_cases=vague, metrics=[vague_handling, relevancy])

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nWhat to look for:")
    print("  Detailed: Groundedness + Usefulness should PASS")
    print("  Vague:    Vague Handling should PASS")
    print("  All:      Relevancy should PASS")
    print("\nSave a screenshot for before/after comparison.")

if __name__ == "__main__":
    print("=" * 60)
    print("FEEDBACK TO ENGINEERING INSIGHTS - DeepEval Evaluation")
    print("=" * 60)
    run_evaluation()