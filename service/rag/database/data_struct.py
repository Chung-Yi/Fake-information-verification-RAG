import os

class SocialMediaData:
    def __init__(self, data, vector):
        self.subject = data["subject"]
        self.vector = vector
        self.content = data["content"]
        self.link = data["permanent-link"]
        self.time = data["post-time"]
        self.name = data["site-name"]
        self.chunk_data = None


