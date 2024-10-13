from transport import Train

previous_state = ""

while True:
    if previous_state != current_state:
        print(current_state)

    previous_state = current_state
