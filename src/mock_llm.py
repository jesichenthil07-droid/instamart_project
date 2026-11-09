
def generate_mock_response(ticket, policy):

    complaint = ticket["complaint"]
    category = ticket["category"]
    policy_name = policy["policy_name"]
    policy_text = policy["policy_text"]

    if policy_name == "No matching policy":
        return (
            "Thank you for contacting Instamart. "
            "We could not find a matching policy. "
            "Your ticket has been assigned to the "
            + ticket["department"] + "."
        )

    response = (
        f"Thank you for contacting Instamart. "
        f"We understand your {category.lower()} issue: "
        f"{complaint}. "
        f"According to our {policy_name}, "
        f"{policy_text} "
        f"Your ticket has been assigned to the "
        f"{ticket['department']} for further assistance."
    )

    return response
