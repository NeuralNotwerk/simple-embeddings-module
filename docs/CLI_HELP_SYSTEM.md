# SEM CLI Help System

The SEM CLI provides comprehensive help at multiple levels to ensure users can easily discover and understand all available functionality.

## Help Access Methods

### 1. Main CLI Help
```bash
sem-cli --help
sem-cli -h
```
Shows overview of all commands with basic examples.

### 2. Command-Specific Help
```bash
sem-cli <command> --help
```
Available for all commands:
- `sem-cli init --help`
- `sem-cli add --help`
- `sem-cli search --help`
- `sem-cli info --help`
- `sem-cli config --help`
- `sem-cli simple --help`
- `sem-cli help --help`

### 3. Interactive Help Command
```bash
sem-cli help                    # General help overview
sem-cli help <command>          # Specific command help
```

Examples:
- `sem-cli help` - Shows all available commands
- `sem-cli help simple` - Shows contextual help for simple interface
- `sem-cli help init` - Directs to `sem-cli init --help`

## Help Hierarchy

### Level 1: Discovery
**Command:** `sem-cli --help` or `sem-cli help`

**Purpose:** Help users discover what commands are available

**Content:**
- List of all commands with brief descriptions
- Basic usage examples
- Pointer to more detailed help

### Level 2: Command Overview
**Command:** `sem-cli <command> --help`

**Purpose:** Show all options and usage for a specific command

**Content:**
- Complete usage syntax
- All available options with descriptions
- Default values where applicable
- Detailed examples (for simple command)

### Level 3: Contextual Help
**Command:** `sem-cli help simple`

**Purpose:** Provide focused, task-oriented help

**Content:**
- Usage patterns and workflows
- Backend-specific examples
- Common use cases and patterns
- Quick reference for operations

## Simple Command Help Features

### Comprehensive Examples
The simple command help includes:

```bash
sem-cli simple --help
```

**Provides:**
- Complete syntax for all backend/operation combinations
- Default values for all options
- Extensive examples section with:
  - Local backend examples
  - AWS backend examples
  - Common pipeline patterns
  - Custom configuration examples

### Contextual Error Messages
When commands fail due to missing arguments:

```bash
# Missing search query
sem-cli simple local search
# Shows: ❌ Search operation requires --query argument
#        Example: sem-cli simple local search --query 'your search terms'

# Missing text input
sem-cli simple local index
# Shows: ❌ No text to index. Provide text via stdin or --text arguments
#        Examples:
#          echo 'some text' | sem-cli simple local index
#          sem-cli simple local index --text 'document 1' 'document 2'
```

### Progressive Disclosure
Help is structured to guide users from general to specific:

1. **Discovery**: `sem-cli --help` shows simple command exists
2. **Overview**: `sem-cli simple --help` shows all options and examples
3. **Context**: `sem-cli help simple` shows workflow-focused help
4. **Validation**: Error messages provide specific examples for fixes

## Help System Testing

### All Help Methods Work
✅ `sem-cli --help` - Main help
✅ `sem-cli help` - Interactive help overview
✅ `sem-cli help simple` - Contextual simple help
✅ `sem-cli simple --help` - Complete simple command help
✅ `sem-cli init --help` - Traditional command help
✅ Error messages with examples for missing arguments

### Incomplete Command Handling
✅ `sem-cli simple` - Shows usage and missing arguments
✅ `sem-cli simple local` - Shows missing operation
✅ `sem-cli simple local search` - Shows missing --query with example
✅ `sem-cli simple local index` - Shows missing input with examples

### Help Discoverability
✅ Main help mentions simple command with examples
✅ Help command provides overview and pointers
✅ Simple help includes comprehensive examples
✅ Error messages include corrective examples

## User Journey Examples

### New User Discovery
```bash
# User starts here
sem-cli --help
# Sees simple command examples, tries:
echo "test" | sem-cli simple local index
# Success! Then tries:
sem-cli simple local search --query "test"
# Success! User is productive immediately
```

### Advanced User Reference
```bash
# User needs AWS options
sem-cli simple --help
# Sees all AWS options with defaults and examples
sem-cli simple aws index --bucket my-bucket --region us-west-2 --text "content"
# Success with custom configuration
```

### Error Recovery
```bash
# User makes mistake
sem-cli simple aws search
# Gets helpful error with exact fix:
# ❌ Search operation requires --query argument
# Example: sem-cli simple aws search --query 'your search terms' --bucket my-bucket
sem-cli simple aws search --query "deployment" --bucket my-bucket
# Success!
```

## Help System Benefits

### 1. **Multiple Access Patterns**
- Traditional `--help` flags
- Interactive `help` command
- Contextual error messages
- Progressive disclosure

### 2. **Task-Oriented Examples**
- Real-world usage patterns
- Pipeline integration examples
- Backend-specific guidance
- Common workflow demonstrations

### 3. **Error Prevention**
- Clear argument requirements
- Default value documentation
- Example-driven error messages
- Validation with helpful suggestions

### 4. **Discoverability**
- Commands are easy to find
- Examples show immediate value
- Help leads to help
- Error messages teach correct usage

The help system ensures that users can be productive immediately while also providing comprehensive reference material for advanced usage.
