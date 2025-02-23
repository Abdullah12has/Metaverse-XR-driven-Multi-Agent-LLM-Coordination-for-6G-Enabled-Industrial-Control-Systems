import time
import logging

class RaftNode:
    def __init__(self, pod_ip):
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.state = 'follower'
        self.leader_address = f"{pod_ip}:50051"
        self.pod_ip = pod_ip
        self.election_timeout = 1  # Seconds
        logging.basicConfig(level=logging.INFO)

    def start_election(self):
        if self.state == 'follower':
            self.state = 'candidate'
            self.current_term += 1
            self.voted_for = self.pod_ip
            # Simplified election - assume we win immediately
            self.become_leader()

    def become_leader(self):
        self.state = 'leader'
        self.leader_address = f"{self.pod_ip}:50051"
        logging.info(f"Elected leader at {self.leader_address} (Term {self.current_term})")

    def is_leader(self):
        return self.state == 'leader'

    def request_vote(self, candidate_term, candidate_id):
        if candidate_term > self.current_term:
            self.current_term = candidate_term
            self.voted_for = candidate_id
            return True
        return False

    def append_entries(self, term, leader_id, entries):
        if term >= self.current_term:
            self.current_term = term
            self.log.extend(entries)
            return True
        return False