walker interact {
    can return_message with `root entry {
        report {
            "response": "Hello, world!"
        };
    }
}

walker interact_with_body {
    has name: str;

    can return_message with `root entry {
        report {
            "response": "Hello, " + self.name + "!"
        };
    }
}

walker new_interact {
    has name: str;
    has time: bool;

    can return_message with `root entry {
        if self.time {
            report {
                "response": "Good morning, " + self.name + "!"
            };
        } else {
            report {
                "response": "Good evening, " + self.name + "!"
            };
        }
    }
}

walker new_new_interact {
    has a: int;
    has b: int;

    can return_message with `root entry {
        report {
            "response": "difference = " + str(self.a - self.b)
        };
    }
}