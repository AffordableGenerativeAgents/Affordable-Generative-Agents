

actions_description = {
    "Walk": "Walks to a room or object (character is not sitting, object is not grabbed)",
    "Sit": "Sit on an object (character is not sitting, character is close to object)",
    "StandUp": "Stand Up (character state is sitting)",
    "Grab": "Grab an object (obj property is GRABBABLE except water, character is close to obj)",
    "Open": "Open an object (obj property is CAN_OPEN and state is CLOSED, character is close to obj)",
    "Close": "Close an object (obj property is CAN_OPEN and state is OPEN, character is close to obj)",
    "Put": "Put an obj1 on obj2 (character hold obj1, character is close to obj2)",
    "PutIn": "Put an obj1 inside obj2 (character hold obj1, character is close to obj2, obj2 is not CLOSED)",
    "SwitchOn": "Turn an object on (object has property HAS_SWITCH, object state is OFF, character is close to object)",
    "SwitchOff": "Turn an object off (object has property HAS_SWITCH, object state is ON, character is close to object)",
    "Drink": "Drink from an object (object property is RECIPIENT, character is close to object)",
    "Touch": "Touch an object (character is close to object)",
    "LookAt": "Look at an object (character is facing object)"
}

actions_instruction = {
    "Walk": "walk obj",
    "Sit": "sit obj",
    "StandUp": "standup",
    "Grab": "grab obj",
    "Open": "open obj",
    "Close": "close obj",
    "Put": "putback obj1 obj2",
    "PutIn": "putin obj1 obj2",
    "SwitchOn": "switchon obj",
    "SwitchOff": "switchoff obj",
    "Drink": "drink obj",
    "Touch": "touch obj",
    "LookAt": "lookat obj"
}

instruction_to_action_name = {
    "walk": "Walk",
    "sit": "Sit",
    "standup": "StandUp",
    "grab": "Grab",
    "open": "Open",
    "close": "Close",
    "putback": "Put",
    "putin": "PutIn",
    "switchon": "SwitchOn",
    "switchoff": "SwitchOff",
    "drink": "Drink",
    "touch": "Touch",
    "lookat": "LookAt"
}