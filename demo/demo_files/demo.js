/**
 * Modern JavaScript demo showcasing ES6+ features and patterns
 */

// ES6 Classes and inheritance
class EventEmitter {
    constructor() {
        this.events = new Map();
    }

    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(callback);
        return this;
    }

    emit(event, ...args) {
        const callbacks = this.events.get(event);
        if (callbacks) {
            callbacks.forEach(callback => callback(...args));
        }
        return this;
    }

    off(event, callback) {
        const callbacks = this.events.get(event);
        if (callbacks) {
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
        return this;
    }
}

// Modern async/await patterns
class APIClient extends EventEmitter {
    constructor(baseURL, options = {}) {
        super();
        this.baseURL = baseURL;
        this.timeout = options.timeout || 5000;
        this.retries = options.retries || 3;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        for (let attempt = 1; attempt <= this.retries; attempt++) {
            try {
                this.emit('request:start', { url, attempt });

                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);

                const response = await fetch(url, {
                    ...config,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                this.emit('request:success', { url, data });
                return data;

            } catch (error) {
                this.emit('request:error', { url, error, attempt });

                if (attempt === this.retries) {
                    throw error;
                }

                // Exponential backoff
                await this.delay(Math.pow(2, attempt) * 1000);
            }
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Functional programming patterns
const DataProcessor = {
    // Higher-order functions
    pipe: (...functions) => (value) => functions.reduce((acc, fn) => fn(acc), value),

    compose: (...functions) => (value) => functions.reduceRight((acc, fn) => fn(acc), value),

    // Array processing utilities
    chunk: (array, size) => {
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    },

    groupBy: (array, keyFn) => {
        return array.reduce((groups, item) => {
            const key = keyFn(item);
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(item);
            return groups;
        }, {});
    },

    // Data transformation pipeline
    processUserData: function(users) {
        return this.pipe(
            data => data.filter(user => user.active),
            data => data.map(user => ({
                ...user,
                fullName: `${user.firstName} ${user.lastName}`,
                age: this.calculateAge(user.birthDate)
            })),
            data => this.groupBy(data, user => user.department),
            data => Object.entries(data).map(([dept, users]) => ({
                department: dept,
                count: users.length,
                averageAge: users.reduce((sum, u) => sum + u.age, 0) / users.length,
                users: users.sort((a, b) => a.fullName.localeCompare(b.fullName))
            }))
        )(users);
    },

    calculateAge: (birthDate) => {
        const today = new Date();
        const birth = new Date(birthDate);
        let age = today.getFullYear() - birth.getFullYear();
        const monthDiff = today.getMonth() - birth.getMonth();

        if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
            age--;
        }

        return age;
    }
};

// Modern module pattern with async initialization
class DatabaseManager {
    #connection = null;
    #isConnected = false;

    constructor(config) {
        this.config = config;
        this.queryCache = new Map();
    }

    async connect() {
        if (this.#isConnected) {
            return this.#connection;
        }

        try {
            // Simulated database connection
            this.#connection = await this.#establishConnection();
            this.#isConnected = true;
            console.log('Database connected successfully');
            return this.#connection;
        } catch (error) {
            console.error('Database connection failed:', error);
            throw error;
        }
    }

    async #establishConnection() {
        // Simulate async connection process
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (Math.random() > 0.1) { // 90% success rate
                    resolve({ id: Date.now(), status: 'connected' });
                } else {
                    reject(new Error('Connection timeout'));
                }
            }, 100);
        });
    }

    async query(sql, params = []) {
        if (!this.#isConnected) {
            await this.connect();
        }

        const cacheKey = `${sql}:${JSON.stringify(params)}`;

        if (this.queryCache.has(cacheKey)) {
            console.log('Cache hit for query:', sql);
            return this.queryCache.get(cacheKey);
        }

        // Simulate query execution
        const result = await new Promise(resolve => {
            setTimeout(() => {
                resolve({
                    rows: [{ id: 1, name: 'Sample Data' }],
                    rowCount: 1,
                    executionTime: Math.random() * 100
                });
            }, 50);
        });

        this.queryCache.set(cacheKey, result);
        return result;
    }
}

// Demo execution
async function runDemo() {
    console.log('🚀 Starting JavaScript Demo...\n');

    // API Client demo
    const api = new APIClient('https://jsonplaceholder.typicode.com');

    api.on('request:start', ({ url, attempt }) => {
        console.log(`📡 Request ${attempt}: ${url}`);
    });

    api.on('request:success', ({ data }) => {
        console.log('✅ Request successful, received data');
    });

    try {
        const userData = await api.request('/users/1');
        console.log('User data:', userData.name);
    } catch (error) {
        console.log('❌ API request failed:', error.message);
    }

    // Data processing demo
    const sampleUsers = [
        { firstName: 'John', lastName: 'Doe', birthDate: '1990-05-15', department: 'Engineering', active: true },
        { firstName: 'Jane', lastName: 'Smith', birthDate: '1985-08-22', department: 'Marketing', active: true },
        { firstName: 'Bob', lastName: 'Johnson', birthDate: '1992-12-03', department: 'Engineering', active: false },
        { firstName: 'Alice', lastName: 'Brown', birthDate: '1988-03-10', department: 'Sales', active: true }
    ];

    const processedData = DataProcessor.processUserData(sampleUsers);
    console.log('\n📊 Processed user data by department:');
    processedData.forEach(dept => {
        console.log(`${dept.department}: ${dept.count} users, avg age ${dept.averageAge.toFixed(1)}`);
    });

    // Database demo
    const db = new DatabaseManager({ host: 'localhost', port: 5432 });
    const result = await db.query('SELECT * FROM users WHERE active = ?', [true]);
    console.log('\n💾 Database query result:', result.rowCount, 'rows');

    console.log('\n🎉 JavaScript demo completed!');
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EventEmitter, APIClient, DataProcessor, DatabaseManager, runDemo };
} else {
    // Browser environment
    window.JSDemo = { EventEmitter, APIClient, DataProcessor, DatabaseManager, runDemo };
}

// Auto-run if this is the main module
if (typeof require !== 'undefined' && require.main === module) {
    runDemo().catch(console.error);
}
