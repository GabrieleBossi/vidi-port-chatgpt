"""
ChatGPT

This module provides an example flow of a ChatGPT data donation study

Assumptions:
It handles DDPs in the english language with filetype JSON.
"""
import logging
from typing import Tuple

import pandas as pd
import numpy as np

import port.api.props as props
import port.api.d3i_props as d3i_props
import port.helpers.extraction_helpers as eh
import port.helpers.validate as validate
from port.platforms.flow_builder import FlowBuilder

from port.helpers.validate import (
    DDPCategory,
    DDPFiletype,
    Language,
)

logger = logging.getLogger(__name__)

DDP_CATEGORIES = [
    DDPCategory(
        id="json",
        ddp_filetype=DDPFiletype.JSON,
        language=Language.EN,
        known_files=[
            "chat.html", 
            "conversations.json",
            "message_feedback.json",
            "model_comparisons.json",
            "user.json"
        ]
    )
]

def extract_conversations(chatgpt_zip: str) -> pd.DataFrame:

    b = eh.extract_file_from_zip(chatgpt_zip, "conversations.json")
    conversations = eh.read_json_from_bytes(b)

    datapoints = []
    out = pd.DataFrame()

    try:
        for conversation in conversations:
            title = conversation["title"]
            conversation_id = conversation["conversation_id"]
            i = 0
            for _, turn in conversation["mapping"].items():

                denested_d = eh.dict_denester(turn)
                is_hidden = eh.find_item(denested_d, "is_visually_hidden_from_conversation")
                content_type = eh.find_item(denested_d, "content_type")
                role = eh.find_item(denested_d, "role")
                if (content_type != "text") or (is_hidden == "True") or (role not in ["user", "assistant"]):
                    continue
                message = "".join(eh.find_items(denested_d, "part"))
                model = eh.find_item(denested_d, "-model_slug")
                time = eh.epoch_to_iso(eh.find_item(denested_d, "create_time"))

                datapoint = {
                    "conversation title": title,
                    "role": role,
                    "message": message,
                    "model": model,
                    "time": time,
                    "conversation_id": conversation_id,
                    "is_first": True if i < 2 else False  # Label first qa pair
                }
                datapoints.append(datapoint)
                i += 1 

        out = pd.DataFrame(datapoints)

    except Exception as e:
        logger.error("Data extraction error: %s", e)
        
    return out


def conversations_to_df(conversations: pd.DataFrame) -> pd.DataFrame:

    df = conversations.copy()
    df = df[[
        "conversation title", 
        "role",
        "message",
        "model",
        "time",
        "conversation_id",
        "is_first"
    ]]
    
    return df



def extraction(chatgpt_zip: str) -> list[d3i_props.PropsUIPromptConsentFormTableViz]:
    """
    Add your table definitions below in the list
    """
    tables = [
        d3i_props.PropsUIPromptConsentFormTableViz(
            id="chatgpt_conversations",
            data_frame=conversations_to_df(extract_conversations(chatgpt_zip)),
            title=props.Translatable({
                "en": "Your conversations with ChatGPT",
                "nl": "Uw gesprekken met ChatGPT"
            }),
            description=props.Translatable({
                "en": "In this table you find your conversations with ChatGPT sorted by time. Below, you find a wordcloud, where the size of the words represents how frequent these words have been used in the conversations.", 
                "nl": "In this table you find your conversations with ChatGPT sorted by time. Below, you find a wordcloud, where the size of the words represents how frequent these words have been used in the conversations.", 
            }),
            visualizations=[
                {
                    "title": {
                        "en": "Your messages in a wordcloud", 
                        "nl": "Your messages in a wordcloud"
                    },
                    "type": "wordcloud",
                    "textColumn": "message",
                    "tokenize": True,
                }
            ]
        ),
    ]

    tables_to_render = [table for table in tables if not table.data_frame.empty]

    return tables_to_render


def select_three_qas(donated_data: list[dict])  -> list[Tuple[str, str]]:
    """
    Code to extract first, middle and last message sent by the user and corresponding answer by ChatGPT
    The extra effort is made here to make sure the answers is actually a follow up of the question 
    and to make sure the question is the first in the conversation
    """

    # Only consider conversations where the first qa pair wasn't deleted
    conversations = pd.DataFrame(donated_data)
    conversations = conversations[conversations.is_first == "true"].groupby("conversation_id", as_index=False).filter(lambda x: len(x) == 2)

    # Select first, last and middle conversation if possible   
    conversation_ids = conversations['conversation_id'].unique()
    if len(conversation_ids) == 0:
        indexes = []
    elif len(conversation_ids) == 1:
        indexes = [0]
    elif len(conversation_ids) == 2:
        indexes = [0, 1]
    else:
        indexes = [0, len(conversation_ids)//2, -1]
    selected_ids = [conversation_ids[i] for i in indexes]
    selected_conversations = conversations[conversations['conversation_id'].isin(selected_ids)]
    
    questions_and_answers = []
    for conversation_id, group in selected_conversations.groupby('conversation_id'):
        questions_and_answers.append(group["message"].tolist())

    return questions_and_answers


# Random question questionnaire

# Question measuring trust in answer provided by ChatGPT

Q1 = props.Translatable(
    {
        "en": "To what extent do you trust the answer provided by ChatGPT?",
        "nl": "Hoeveel vertrouwen heeft u in het antwoord van ChatGPT?"
    })
Q1_CHOICES = [
    props.Translatable(
        {
            "en": "1. I do not trust it at all", 
            "nl": "1. Helemaal geen vertrouwen"
        }),
    props.Translatable(
        {
            "en": "2", 
             "nl": "2"
        }),
    props.Translatable(
        {
            "en": "3", 
            "nl": "3"
        }),
    props.Translatable(
        {
            "en": "4",
             "nl": "4"
         }),
    props.Translatable(
        {
            "en": "5",
            "nl": "5"
         }),
    props.Translatable(
        {
            "en": "6",
             "nl": "6"
         }),
    props.Translatable({
        "en": "7. I trust it completely", 
        "nl": "7. Volledig vertrouwen"
    })
]

# Question measuring privacy

Q2 = props.Translatable(
    {
        "en": "The information in this conversation is:",
        "nl": "De informatioe in dit gesprek is:"
    })
Q2_CHOICES = [
    props.Translatable(
        {
            "en": "1. Not at all personal about me", 
            "nl": "1. Helemaal niet persoonlijk over mij"
        }),
    props.Translatable(
        {
            "en": "2", 
             "nl": "2"
        }),
    props.Translatable(
        {
            "en": "3", 
            "nl": "3"
        }),
    props.Translatable(
        {
            "en": "4",
             "nl": "4"
         }),
    props.Translatable(
        {
            "en": "5",
            "nl": "5"
         }),
    props.Translatable(
        {
            "en": "6",
             "nl": "6"
         }),
    props.Translatable({
        "en": "7. Highly personal about me", 
        "nl": "7. Heel persoonlijk over mij"
    })
]

# Question measuring use type

Q3 = props.Translatable(
    {
        "en": "What is this conversation about?",
        "nl": "Waar gaat dit gesprek over?"
    })
Q3_CHOICES = [
    props.Translatable(
        {
            "en": "1. Help with work or school", 
            "nl": "1. Hulp bij werk of school"
        }),
    props.Translatable(
        {
            "en": "2. Writing texts or improving them", 
             "nl": "2. Teksten schrijven of beter maken"
        }),
    props.Translatable(
        {
            "en": "3. Entertainment or doing something fun", 
            "nl": "3. Amusement of iets leuks doen"
        }),
    props.Translatable(
        {
            "en": "4. Coming up with creative ideas or new things",
             "nl": "4. Creatieve ideeën of nieuwe dingen bedenken"
         }),
    props.Translatable(
        {
            "en": "5. Help to learn something new or understand something difficult better",
            "nl": "5. Hulp om iets nieuws te leren of iets moeilijks beter te snappen"
         }),
    props.Translatable(
        {
            "en": "6. Looking up information or answering questions",
             "nl": "6. Informatie zoeken of vragen beantwoorden"
         }),
    props.Translatable({
            "en": "7. News and current events", 
            "nl": "7. Nieuws en actualiteiten"
        }), 
    props.Translatable({
            "en": "8. Just talking or looking for company", 
             "nl": "8. Gewoon praten of gezelschap zoeken"
        }), 
    props.Translatable({
            "en": "9. Talking about personal questions or sensitive topics", 
            "nl": "9. Persoonlijke vragen of gevoelige onderwerpen bespreken"
        }), 
    props.Translatable({
            "en": "10. Help with daily things, like cooking, traveling, or other practical tasks", 
             "nl": "10. Hulp bij dagelijkse dingen, zoals koken, reizen of andere praktische zaken"
        }), 
    props.Translatable({
        "en": "11. Something else", 
        "nl": "11. Iets anders"
    })
]

#def render_questionnaire(question: str, answer: str):
#    questions = [
#        props.PropsUIQuestionMultipleChoice(question=Q1, id=1, choices=Q1_CHOICES),
#    ]
#
#    description = props.Translatable(
#        {
#            "en": "Below you can find the start of a conversation you had with ChatGPT. We would like to ask you a question about it.",
#            "nl": "Hieronder vind u het begin van een gesprek dat u heeft gehad met ChatGPT. We willen u daar een vraag over stellen."
#        })
#    header = props.PropsUIHeader(props.Translatable({"en": "Questionnaire", "nl": "Vragenlijst"}))
#    body = props.PropsUIPromptQuestionnaire(
#        questions=questions, 
#        description=description,
#        questionToChatgpt=question,
#        answerFromChatgpt=answer,
#    )
#    footer = props.PropsUIFooter()
#
#    page = props.PropsUIPageDonation("ASD", header, body, footer)
#    return CommandUIRender(page)


def generate_questionnaire(question: str, answer: str, index: int) -> d3i_props.PropsUIPromptQuestionnaire:
    """
    Administer a basic questionnaire in Port.
    This function generates a prompt which can be rendered with render_page().
    The questionnaire demonstrates all currently implemented question types.
    In the current implementation, all questions are optional.
    You can build in logic by:
    - Chaining questionnaires together
    - Using extracted data in your questionnaires
    Usage:
        prompt = generate_questionnaire()
        results = yield render_page(header_text, prompt)
        
    The results.value contains a JSON string with question answers that 
    can then be donated with donate().
    """
    
    ordinals_en = {1: "first", 2: "second", 3: "third"}
    ordinals_nl = {1: "eerste", 2: "tweede", 3: "derde"}
    
    ordinal_en = ordinals_en.get(index, f"{index}th")
    ordinal_nl = ordinals_nl.get(index, f"{index}e")
    
    # Adjust article for Dutch
    article_nl = "een" if index == 1 else "een"
    
    questionnaire_description = props.Translatable({
        "en": f"Below you can find the start of a {ordinal_en} conversation you had with ChatGPT. We would like to ask you a few questions about it.",
        "nl": f"Hieronder vind u het begin van {article_nl} {ordinal_nl} gesprek dat u heeft gehad met ChatGPT. We willen u daar een paar vragen over stellen."
    })
    
    multiple_choice_trust = d3i_props.PropsUIQuestionMultipleChoice(
        id=f"{index}-trust",
        question=Q1,
        choices=Q1_CHOICES,
    )

    multiple_choice_privacy = d3i_props.PropsUIQuestionMultipleChoice(
        id=f"{index}-privacy",
        question=Q2,
        choices=Q2_CHOICES,
    )

    multiple_choice_usetype = d3i_props.PropsUIQuestionMultipleChoice(
        id=f"{index}-usetype",
        question=Q3,
        choices=Q3_CHOICES,
    )
    
    return d3i_props.PropsUIPromptQuestionnaire(
        description=questionnaire_description,
        questions=[
            multiple_choice_trust,
            multiple_choice_privacy,
            multiple_choice_usetype
        ],
        questionToChatgpt=question,
        answerFromChatgpt=answer,
    )


class ChatGPTFlow(FlowBuilder):
    def __init__(self, session_id: int):
        super().__init__(session_id, "ChatGPT")
        
    def validate_file(self, file):
        return validate.validate_zip(DDP_CATEGORIES, file)
        
    def extract_data(self, file_value, validation):
        return extraction(file_value)


def process(session_id):
    flow = ChatGPTFlow(session_id)
    return flow.start_flow()
