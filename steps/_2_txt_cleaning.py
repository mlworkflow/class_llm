import pandas as pd
from zenml import step
import re
import string
from typing import Annotated
from pythelpers.ml.text_preprocessing import (
    expand_contractions,
    strip_all_entities,
    filter_chars,
    remove_mult_spaces,
    remove_numbers,
    lemmatize,
    remove_short_words,
    replace_elongated_words,
    remove_repeated_punctuation,
    remove_extra_whitespace,
    remove_url_shorteners,
    remove_short_tickets,
)


@step
def txt_cleaning(df: pd.DataFrame) -> Annotated[pd.DataFrame, "df"]:
    """Handles missing values using MissingValueHandler and the specified strategy."""

    df['text_clean'] = [clean_ticket(ticket) for ticket in df['body']]
    return df



# Function to call all the cleaning functions in the correct order
def clean_ticket(ticket):
    ticket = expand_contractions(ticket)
    ticket = strip_all_entities(ticket)
    ticket = filter_chars(ticket)
    ticket = remove_mult_spaces(ticket)
    ticket = remove_numbers(ticket)
    ticket = lemmatize(ticket)
    ticket = remove_short_words(ticket)
    ticket = replace_elongated_words(ticket)
    ticket = remove_repeated_punctuation(ticket)
    ticket = remove_extra_whitespace(ticket)
    ticket = remove_url_shorteners(ticket)
    ticket = remove_short_tickets(ticket)
    ticket = ' '.join(ticket.split())  # Remove multiple spaces between words
    return ticket