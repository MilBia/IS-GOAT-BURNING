import signal


class SignalHandler:
    """
    Handle proper exit on quiting quieting CTRL+C
    """

    KEEP_PROCESSING = True

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("Terminating processes.")
        self.KEEP_PROCESSING = False
