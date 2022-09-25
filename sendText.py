import requests


def send(message, number="5057300932"):
    resp = requests.post(
        "https://textbelt.com/text",
        {
            "phone": number,
            "message": message,
            "key": "a9e607286caf52e169910d23aea3bfad64d465b2OM2dy54HQxkLRLsDPsZXANBLX",
        },
    )
    print(resp.json())
