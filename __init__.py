import random

welcome_text = [
    "I am Noa Roggendorff",
    "You have done well to reach this place", 
    "Let us begin the training process"
]

length = max(len(text) for text in welcome_text) + 4

welcome_messages = ''.join([
    f"\t[ {'-' * ((length - len(text)) // 2)} {text} {'-' * ((length - len(text) + 1) // 2)} ]\n"
    for text in welcome_text
])

llama_one = random.choice([True, False])

dashes = "-" * length

pos = random.randint(0, len(dashes))
llama_dashes = dashes[:pos] + "🦙" + dashes[pos:]

llama_text = f"\t[ {llama_dashes} ]"
plain_text = f"\t[ -{dashes}- ]"

print(llama_text if llama_one else plain_text)
print(welcome_messages.rstrip())
print(llama_text if not llama_one else plain_text)
