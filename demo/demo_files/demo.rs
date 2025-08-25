use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait Database {
    fn connect(&mut self) -> Result<(), String>;
    fn execute(&self, query: &str) -> Result<Vec<String>, String>;
}

pub struct PostgresDB {
    connection_string: String,
    connected: bool,
}

impl PostgresDB {
    pub fn new(conn_str: String) -> Self {
        Self {
            connection_string: conn_str,
            connected: false,
        }
    }
}

impl Database for PostgresDB {
    fn connect(&mut self) -> Result<(), String> {
        println!("Connecting to: {}", self.connection_string);
        self.connected = true;
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<Vec<String>, String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }
        Ok(vec![format!("Result for: {}", query)])
    }
}

#[derive(Debug, Clone)]
pub struct User {
    pub id: u32,
    pub name: String,
    pub email: String,
}

pub struct UserService {
    users: Arc<Mutex<HashMap<u32, User>>>,
    db: Box<dyn Database + Send>,
}

impl UserService {
    pub fn new(db: Box<dyn Database + Send>) -> Self {
        Self {
            users: Arc::new(Mutex::new(HashMap::new())),
            db,
        }
    }

    pub fn add_user(&self, user: User) -> Result<(), String> {
        let mut users = self.users.lock().unwrap();
        users.insert(user.id, user);
        Ok(())
    }

    pub fn get_user(&self, id: u32) -> Option<User> {
        let users = self.users.lock().unwrap();
        users.get(&id).cloned()
    }
}

fn main() {
    let mut db = PostgresDB::new("postgresql://localhost:5432/demo".to_string());
    db.connect().expect("Failed to connect");

    let service = UserService::new(Box::new(db));

    let user = User {
        id: 1,
        name: "Alice".to_string(),
        email: "alice@example.com".to_string(),
    };

    service.add_user(user).expect("Failed to add user");

    if let Some(retrieved_user) = service.get_user(1) {
        println!("Retrieved user: {:?}", retrieved_user);
    }
}
