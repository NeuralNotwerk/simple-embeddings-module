"""Sample code files for the ultimate chunking demo - Part 1: Core Languages"""

# Python - Object-oriented calculator
PYTHON_CODE = '''#!/usr/bin/env python3
"""Advanced calculator with history and operations."""

import math
import json
from typing import List, Dict, Optional
from datetime import datetime

class Calculator:
    """A feature-rich calculator with history tracking."""

    def __init__(self, precision: int = 2):
        self.precision = precision
        self.history: List[Dict] = []
        self.memory: float = 0.0

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = round(a + b, self.precision)
        self._record_operation("add", [a, b], result)
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = round(a * b, self.precision)
        self._record_operation("multiply", [a, b], result)
        return result

    def power(self, base: float, exponent: float) -> float:
        """Calculate power."""
        result = round(math.pow(base, exponent), self.precision)
        self._record_operation("power", [base, exponent], result)
        return result

    def _record_operation(self, op: str, operands: List[float], result: float):
        """Record operation in history."""
        self.history.append({
            "operation": op,
            "operands": operands,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self) -> List[Dict]:
        """Get calculation history."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()

def factorial(n: int) -> int:
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

async def fetch_math_constants() -> Dict[str, float]:
    """Fetch mathematical constants."""
    return {
        "pi": math.pi,
        "e": math.e,
        "golden_ratio": (1 + math.sqrt(5)) / 2
    }

if __name__ == "__main__":
    calc = Calculator(precision=3)
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"4 * 7 = {calc.multiply(4, 7)}")
    print(f"2^8 = {calc.power(2, 8)}")
    print(f"5! = {factorial(5)}")
'''

# JavaScript - Express.js web server
JAVASCRIPT_CODE = '''// Modern Express.js API server with middleware
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware setup
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

class UserService {
    constructor(database) {
        this.db = database;
        this.users = new Map();
        this.initializeDefaults();
    }

    initializeDefaults() {
        this.users.set('admin', {
            id: 'admin',
            name: 'Administrator',
            role: 'admin',
            created: new Date()
        });
    }

    async createUser(userData) {
        const user = {
            id: this.generateId(),
            ...userData,
            created: new Date(),
            lastLogin: null
        };

        this.users.set(user.id, user);
        await this.db.save('users', user);
        return user;
    }

    async getUserById(id) {
        if (this.users.has(id)) {
            return this.users.get(id);
        }
        return await this.db.findById('users', id);
    }

    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
}

function validateEmail(email) {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}

const hashPassword = async (password) => {
    const bcrypt = require('bcrypt');
    return await bcrypt.hash(password, 12);
};

// API Routes
app.get('/api/users', async (req, res) => {
    try {
        const users = await userService.getAllUsers();
        res.json({ success: true, data: users });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.post('/api/users', async (req, res) => {
    try {
        const { name, email, password } = req.body;

        if (!validateEmail(email)) {
            return res.status(400).json({ success: false, error: 'Invalid email' });
        }

        const hashedPassword = await hashPassword(password);
        const user = await userService.createUser({ name, email, password: hashedPassword });

        res.status(201).json({ success: true, data: user });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Server startup
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app;
'''

# TypeScript - React component with hooks
TYPESCRIPT_CODE = '''// Modern React component with TypeScript and hooks
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';

interface User {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'user' | 'guest';
    created: Date;
    lastLogin?: Date;
}

interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
}

type UserFormData = Omit<User, 'id' | 'created' | 'lastLogin'>;

class UserApiClient {
    private baseUrl: string;
    private token?: string;

    constructor(baseUrl: string, token?: string) {
        this.baseUrl = baseUrl;
        this.token = token;
    }

    async getUsers(): Promise<ApiResponse<User[]>> {
        try {
            const response = await axios.get(`${this.baseUrl}/api/users`, {
                headers: this.getHeaders()
            });
            return response.data;
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async createUser(userData: UserFormData): Promise<ApiResponse<User>> {
        try {
            const response = await axios.post(`${this.baseUrl}/api/users`, userData, {
                headers: this.getHeaders()
            });
            return response.data;
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    private getHeaders(): Record<string, string> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json'
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        return headers;
    }
}

const UserManagement: React.FC = () => {
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const apiClient = useMemo(() => new UserApiClient('/api'), []);

    const fetchUsers = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await apiClient.getUsers();
            if (response.success && response.data) {
                setUsers(response.data);
            } else {
                setError(response.error || 'Failed to fetch users');
            }
        } catch (err) {
            setError('Network error occurred');
        } finally {
            setLoading(false);
        }
    }, [apiClient]);

    useEffect(() => {
        fetchUsers();
    }, [fetchUsers]);

    const handleCreateUser = async (userData: UserFormData) => {
        const response = await apiClient.createUser(userData);
        if (response.success && response.data) {
            setUsers(prev => [...prev, response.data!]);
        }
    };

    if (loading) return <div>Loading users...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div className="user-management">
            <h2>User Management</h2>
            <UserList users={users} />
            <UserForm onSubmit={handleCreateUser} />
        </div>
    );
};

export default UserManagement;
'''

# Continue with more languages...
SAMPLE_FILES = {
    'calculator.py': PYTHON_CODE,
    'server.js': JAVASCRIPT_CODE,
    'components.tsx': TYPESCRIPT_CODE,
}
