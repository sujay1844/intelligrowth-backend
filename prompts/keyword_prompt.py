
keyword_example_prompt = """
[INST]
I have the following document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
meat, beef, eat, eating, emissions, steak, food, health, processed, chicken
[/INST] """

keyword_ins_prompt = """
[INST]
I have the following document:
- [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""

keyword_prompt = keyword_example_prompt + keyword_ins_prompt