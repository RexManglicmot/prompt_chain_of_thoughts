# app/prompt.py

def direct_prompt(context_id, context, question):
    return (
        "You are a clinical researcher.\n\n"
        f"Context (ID:{context_id}):\n{context}\n\n"
        f"Q: {question}\n\n"
        "Instructions:\n"
        "- Output EXACTLY one line with.\n"
        "- The ONLY allowed values are: yes | no | maybe.\n"
        "- Do NOT include reasoning or any other text.\n\n"
        "<FINAL_ANSWER>"
    )

def cot_prompt(context_id, context, question):
    return (
        "You are a clinical researcher.\n\n"
        "Examples:\n"
        "Context:\n"
        "A randomized trial reported aspirin reduced the risk of recurrent myocardial infarction; benefits were consistent across subgroups; comparator was usual care; no contradictory findings were highlighted in the passage.\n"
        "Q: Does aspirin reduce the risk of recurrent heart attack?\n"
        "Final Answer: yes | Reasoning: the passage states aspirin reduced recurrent MI; effect described as reduction not neutrality; result observed across subgroups implying robustness; comparator suggests improvement over standard care; no evidence of harm offsetting benefit is given; language is affirmative rather than uncertain; therefore the correct label is yes\n\n"
        "Context:\n"
        "Vitamin D supplementation improved bone density in older adults; however evidence for fracture reduction was inconclusive according to the passage.\n"
        "Q: Does vitamin D supplementation reduce fractures in older adults?\n"
        "Final Answer: maybe | Reasoning: bone density improvement alone does not guarantee fewer fractures; the context explicitly says fracture evidence is inconclusive; absence of clear benefit means not yes; absence of clear lack of benefit means not no; uncertainty dominates the conclusion; therefore the correct label is maybe\n\n"
        "Context:\n"
"A clinical trial reported that a new cholesterol-lowering drug did not reduce cardiovascular events compared to placebo; the authors concluded there was no benefit.\n"
        "Q: Does the new cholesterol-lowering drug reduce cardiovascular events?\n"
"Final Answer: no | Reasoning: the passage states no reduction was observed; outcomes were the same as placebo; authors concluded no benefit; this rules out yes or maybe; therefore the correct label is no\n\n"
        "---\n\n"
        f"Context (ID:{context_id}):\n{context}\n\n"
        f"Q: {question}\n\n"
        "Instructions:\n"
        "- Output EXACTLY ONE line (no line breaks).\n"
        "- Start with 'Final Answer:' followed by exactly one of: yes | no | maybe.\n"
        "- Then append ' | Reasoning:' and write a LONG explanation of 8–12 micro-steps (about 80–150 words) using ONLY the Context.\n"
        "- Separate micro-steps with '; ' (semicolon + space). No bullets. No extra text before or after the line.\n\n"
        "Final Answer: "
    )
