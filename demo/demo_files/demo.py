#!/usr/bin/env python3
"""
Advanced Python demo showcasing various semantic structures
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class User:
    """User data model with validation"""
    id: int
    name: str
    email: str

    def __post_init__(self):
        if not self.email or '@' not in self.email:
            raise ValueError("Invalid email address")

class DatabaseConnection(ABC):
    """Abstract database connection interface"""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish database connection"""
        pass

    @abstractmethod
    async def execute_query(self, query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        pass

class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database implementation"""

    def __init__(self, host: str, port: int = 5432):
        self.host = host
        self.port = port
        self.connection = None

    async def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            # Simulated connection logic
            await asyncio.sleep(0.1)
            self.connection = f"postgresql://{self.host}:{self.port}"
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def execute_query(self, query: str) -> List[Dict]:
        """Execute PostgreSQL query"""
        if not self.connection:
            raise RuntimeError("Not connected to database")

        # Simulated query execution
        await asyncio.sleep(0.05)
        return [{"result": "success", "query": query}]

def calculate_fibonacci(n: int) -> int:
    """Calculate fibonacci number using memoization"""
    cache = {}

    def fib_helper(num):
        if num in cache:
            return cache[num]
        if num <= 1:
            return num
        cache[num] = fib_helper(num - 1) + fib_helper(num - 2)
        return cache[num]

    return fib_helper(n)

async def process_users_batch(users: List[User], db: DatabaseConnection) -> Dict[str, int]:
    """Process a batch of users asynchronously"""
    results = {"processed": 0, "errors": 0}

    tasks = []
    for user in users:
        task = asyncio.create_task(process_single_user(user, db))
        tasks.append(task)

    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed_tasks:
        if isinstance(result, Exception):
            results["errors"] += 1
        else:
            results["processed"] += 1

    return results

async def process_single_user(user: User, db: DatabaseConnection) -> bool:
    """Process individual user record"""
    query = f"INSERT INTO users (id, name, email) VALUES ({user.id}, '{user.name}', '{user.email}')"
    try:
        await db.execute_query(query)
        return True
    except Exception:
        return False

if __name__ == "__main__":
    # Demo execution
    async def main():
        db = PostgreSQLConnection("localhost")
        await db.connect()

        users = [
            User(1, "Alice Johnson", "alice@example.com"),
            User(2, "Bob Smith", "bob@example.com"),
            User(3, "Carol Davis", "carol@example.com")
        ]

        results = await process_users_batch(users, db)
        print(f"Processing complete: {results}")

        fib_result = calculate_fibonacci(10)
        print(f"Fibonacci(10) = {fib_result}")

    asyncio.run(main())
