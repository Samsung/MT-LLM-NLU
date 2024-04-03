import re

def annotate_with_XML(sentence, BIO):

    sentence_splitted = sentence.strip().split()
    BIO_splitted = BIO.strip().split()
    if len(BIO_splitted) != len(sentence_splitted):
        output_json = {"ERROR" : "len(BIO_splitted) != len(sentence_splitted)"}
        return output_json

    annotated_sentence = []
    slot_ordinal_number = 0
    number_letter_mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5:"f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
                13: "n", 14: "o", 15: "p", 16: "r", 17: "s"}
    slot_letter_mapping = {}
    slot_role_mapping = {}
    slot_signal_type_mapping = {}
    pdss_mapping = {}

    for idx, (token, BIO_token) in enumerate(zip(sentence_splitted, BIO_splitted)):
        if BIO_token == "o":
            annotated_sentence.append(token)
        else:
            BIO_token_start, slot_name = BIO_token.split("-")
            if idx+1 < len(BIO_splitted):
                next_BIO_token = BIO_splitted[idx+1]
            else:
                next_BIO_token = "o"

            tag = f"<{number_letter_mapping[slot_ordinal_number]}>"
            # Put opening XML tag
            if BIO_token_start in ("u", "b"):
                annotated_sentence.append(tag)
                slot_letter_mapping[number_letter_mapping[slot_ordinal_number]] = slot_name
                
            annotated_sentence.append(token)

            # Add closing XML tag
            if BIO_token_start == "u" or \
                    BIO_token_start in ("b", "i") and not next_BIO_token.startswith(("i-", "l-")) or \
                    BIO_token_start == "l":
                annotated_sentence.append(tag)
                slot_ordinal_number += 1

    return {
        "input_sentence" :sentence,
        "input_BIO" : BIO,
        "annotated_sentence": " ".join(annotated_sentence),
        "slot_mapping" : slot_letter_mapping}


def remove_incorrect_tags(sentence, allowed_tags):

    tags_to_remove = []
    # Remove tag if it is not in allowed_tags parameter
    for tag_letter in set(re.findall('(?<=<)[a-s]+(?=>)', sentence)):
        if tag_letter not in allowed_tags:
            # sentence = re.sub(fr'<{tag_letter}>', '', sentence)
            tags_to_remove.append(tag_letter)


    # Remove tag if number of occurrences is not 2 or tag value is empty
    for tag in allowed_tags:
        found_tags = re.findall(fr'<{tag}>', sentence)
        if len(found_tags) != 2 or \
                re.search(fr'<{tag}>\s*<{tag}>', sentence):
            #sentence = re.sub(fr'<{tag}>', '', sentence)
            tags_to_remove.append(tag)

    # Remove overlapping tags
    for tag in allowed_tags:
        # if re.search(fr'<{tag}>([^<]+<[^{tag}]+>)+[^<]+<{tag}>', sentence):
        if re.search(fr"<{tag}>.*(<[^{tag}]>).*<{tag}>", sentence):
            # sentence = re.sub(fr'<{tag}>', '', sentence)
            tags_to_remove.append(tag)
    for tag in tags_to_remove:
        sentence = re.sub(fr'<{tag}>', '', sentence)

    sentence = re.sub(r" {2,}", " ", sentence)
    return sentence.strip()

def decode_translation(translated_sentence, output_json):
    slot_mapping = output_json["slot_mapping"]
    translated_sentence = remove_incorrect_tags(translated_sentence, slot_mapping.keys())

    # If token ends with punctuation add space before punctuation character
    translated_sentence = re.sub(r"(?<=\S)([.,?!])(?=\s|$)", r" \1", translated_sentence)
    # Normalize spaces, insert spaces around tags
    translated_sentence = re.sub(r"(?<=\S)(<[a-z]>)", r" \1", translated_sentence)
    translated_sentence = re.sub(r"(<[a-z]>)(?=[^ ])", r"\1 ", translated_sentence)
    translated_sentence = re.sub(r" {2,}", " ", translated_sentence)
    translated_sentence = translated_sentence.strip()

    tokens = []
    BIO = []
    role = []
    signal_type = []
    pdss = []

    tag = None
    first_token = False
    for word in translated_sentence.split():
        # If current word equals latest seen tag then it is closing tag
        if tag == word:
            tag = None
            continue
        # Set tag and tag_letter if word matches regex
        if re.match(r'<[a-s]>', word):
            tag = word
            tag_letter = word.strip("<>")
            first_token = True
            continue
        if tag:
            tokens.append(word)
            if first_token:
                prefix = "b-"
                first_token = False
            else:
                prefix = "i-"
            # `slot_mapping[tag_letter] != "o"` is always True at this point
            # (as annotate_with_XML does not store "o" slots in mapping),
            # but keep this as additional check for compatibility
            if slot_mapping[tag_letter] != "o":
                BIO.append(f"{prefix}{slot_mapping[tag_letter]}")
            else:
                BIO.append(slot_mapping[tag_letter])
        else:
            tokens.append(word)
            BIO.append("o")

    return tokens, BIO