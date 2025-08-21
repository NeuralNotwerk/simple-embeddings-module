import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

public class Demo {
    
    public interface Repository<T, ID> {
        Optional<T> findById(ID id);
        List<T> findAll();
        T save(T entity);
        void deleteById(ID id);
    }
    
    public static class User {
        private final Long id;
        private final String name;
        private final String email;
        
        public User(Long id, String name, String email) {
            this.id = id;
            this.name = name;
            this.email = email;
        }
        
        public Long getId() { return id; }
        public String getName() { return name; }
        public String getEmail() { return email; }
        
        @Override
        public String toString() {
            return String.format("User{id=%d, name='%s', email='%s'}", id, name, email);
        }
    }
    
    public static class UserRepository implements Repository<User, Long> {
        private final Map<Long, User> users = new HashMap<>();
        
        @Override
        public Optional<User> findById(Long id) {
            return Optional.ofNullable(users.get(id));
        }
        
        @Override
        public List<User> findAll() {
            return new ArrayList<>(users.values());
        }
        
        @Override
        public User save(User user) {
            users.put(user.getId(), user);
            return user;
        }
        
        @Override
        public void deleteById(Long id) {
            users.remove(id);
        }
        
        public List<User> findByNameContaining(String name) {
            return users.values().stream()
                    .filter(user -> user.getName().toLowerCase().contains(name.toLowerCase()))
                    .collect(Collectors.toList());
        }
    }
    
    public static class UserService {
        private final UserRepository repository;
        
        public UserService(UserRepository repository) {
            this.repository = repository;
        }
        
        public CompletableFuture<User> createUserAsync(String name, String email) {
            return CompletableFuture.supplyAsync(() -> {
                Long id = System.currentTimeMillis();
                User user = new User(id, name, email);
                return repository.save(user);
            });
        }
        
        public List<User> searchUsers(String query) {
            return repository.findByNameContaining(query);
        }
        
        public Optional<User> getUserById(Long id) {
            return repository.findById(id);
        }
    }
    
    public static void main(String[] args) {
        UserRepository repository = new UserRepository();
        UserService service = new UserService(repository);
        
        // Create users asynchronously
        CompletableFuture<User> user1Future = service.createUserAsync("Alice Johnson", "alice@example.com");
        CompletableFuture<User> user2Future = service.createUserAsync("Bob Smith", "bob@example.com");
        CompletableFuture<User> user3Future = service.createUserAsync("Charlie Brown", "charlie@example.com");
        
        // Wait for all users to be created
        CompletableFuture.allOf(user1Future, user2Future, user3Future).join();
        
        // Search for users
        List<User> searchResults = service.searchUsers("alice");
        System.out.println("Search results for 'alice': " + searchResults);
        
        // Get all users
        List<User> allUsers = repository.findAll();
        System.out.println("All users: " + allUsers);
        
        // Demonstrate stream processing
        String userSummary = allUsers.stream()
                .map(User::getName)
                .collect(Collectors.joining(", ", "Users: [", "]"));
        System.out.println(userSummary);
    }
}
