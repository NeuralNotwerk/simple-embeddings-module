# Documentation Structure Overview

This document explains the new tiered documentation structure for SEM, designed to serve users at different levels of expertise and use cases.

## Documentation Philosophy

The documentation follows a **progressive disclosure** approach:

1. **Discovery** → Help users find what they need
2. **Quick Start** → Get users productive immediately  
3. **Deep Dive** → Provide comprehensive reference material
4. **Examples** → Show real-world usage patterns

## Tier Structure

### Tier 1: Discovery & Quick Start
**Target**: New users, quick evaluation
**Time Investment**: 5-15 minutes

**Files:**
- `README.md` - Project overview with quick examples
- `docs/quickstart.rst` - 5-minute getting started guide
- `docs/installation.rst` - Platform-specific installation

**Purpose:**
- Help users understand what SEM does
- Get semantic search working immediately
- Provide clear next steps

### Tier 2: User Guides  
**Target**: Regular users, specific workflows
**Time Investment**: 15-60 minutes

**Files:**
- `docs/cli-guide.rst` - Complete CLI usage and scripting
- `docs/simple-interface.rst` - Zero-config semantic search
- `docs/python-api.rst` - Programmatic usage (TODO)
- `docs/configuration.rst` - Custom configurations (TODO)
- `docs/backends.rst` - Storage and embedding options (TODO)

**Purpose:**
- Cover specific interfaces comprehensively
- Show workflow patterns and best practices
- Bridge from simple to advanced usage

### Tier 3: Reference Material
**Target**: Power users, integration developers
**Time Investment**: As needed for reference

**Files:**
- `docs/cli-reference.rst` - Complete command reference
- `docs/api-reference.rst` - Complete Python API (TODO)
- `docs/configuration-reference.rst` - All config options (TODO)
- `docs/troubleshooting.rst` - Common issues and solutions (TODO)

**Purpose:**
- Comprehensive option documentation
- Technical specifications
- Troubleshooting and debugging

### Tier 4: Advanced Topics
**Target**: Developers, customization, production
**Time Investment**: 1+ hours for deep understanding

**Files:**
- `docs/architecture.rst` - System design and components (TODO)
- `docs/chunking-strategies.rst` - Text processing approaches (TODO)
- `docs/embedding-providers.rst` - Model integration (TODO)
- `docs/storage-backends.rst` - Persistence options (TODO)
- `docs/performance.rst` - Optimization and scaling (TODO)

**Purpose:**
- Deep technical understanding
- Customization and extension
- Production deployment guidance

### Tier 5: Examples & Patterns
**Target**: All users, specific use cases
**Time Investment**: 10-30 minutes per example

**Files:**
- `docs/examples/index.rst` - Example overview and organization
- `docs/examples/quickstart-examples.rst` - Copy-paste examples
- `docs/examples/cli-examples.rst` - Shell scripting patterns
- `docs/examples/python-examples.rst` - Application integration
- `docs/examples/pipeline-examples.rst` - Automation workflows
- `docs/examples/aws-examples.rst` - Cloud deployment patterns
- `docs/examples/advanced-examples.rst` - Complex use cases

**Purpose:**
- Real-world usage patterns
- Copy-paste solutions
- Best practice demonstrations

## File Format Strategy

### RST vs Markdown

**RST Files** (`.rst`):
- Main documentation content
- Cross-references and linking
- Sphinx integration
- Professional documentation appearance

**Markdown Files** (`.md`):
- README files and GitHub integration
- Quick reference documents
- Standalone guides

### Content Organization

**Each RST file includes:**
- Clear purpose statement
- Prerequisites and assumptions
- Step-by-step instructions
- Complete working examples
- Next steps and related content

**Consistent structure:**
```rst
Title
=====

Brief description and purpose

Prerequisites
-------------

What users need before starting

Main Content
------------

Step-by-step instructions with examples

Advanced Usage
--------------

Optional advanced topics

Troubleshooting
---------------

Common issues and solutions

Next Steps
----------

Where to go from here
```

## Navigation Strategy

### Entry Points

**New Users:**
1. `README.md` → `docs/quickstart.rst` → Choose path based on needs

**Returning Users:**
1. Direct to relevant guide or reference material

**Developers:**
1. `docs/architecture.rst` → Specific technical topics

### Cross-References

**Every document includes:**
- Clear "Next Steps" section
- Links to related content
- References to examples
- Pointers to reference material

**Example navigation flow:**
```
README.md
├── docs/quickstart.rst
│   ├── docs/cli-guide.rst
│   ├── docs/simple-interface.rst
│   └── docs/examples/quickstart-examples.rst
├── docs/installation.rst
└── docs/examples/index.rst
```

## Help System Integration

### CLI Help Hierarchy

The documentation structure mirrors the CLI help system:

**Level 1: Discovery**
- `sem-cli --help` ↔ `README.md`
- `sem-cli help` ↔ `docs/index.rst`

**Level 2: Command Overview**  
- `sem-cli simple --help` ↔ `docs/cli-reference.rst`
- `sem-cli init --help` ↔ `docs/cli-reference.rst`

**Level 3: Contextual Help**
- `sem-cli help simple` ↔ `docs/cli-guide.rst`
- Error messages ↔ `docs/troubleshooting.rst`

### Progressive Disclosure

**Documentation mirrors user journey:**
1. **Discovery**: What is SEM? (README)
2. **Evaluation**: Can I use this? (quickstart)
3. **Learning**: How do I use this? (guides)
4. **Reference**: What are all the options? (reference)
5. **Mastery**: How do I optimize this? (advanced)

## Content Standards

### Code Examples

**All code examples must:**
- Be complete and runnable
- Include expected output
- Show error handling where appropriate
- Use realistic data and scenarios

### Writing Style

**Tone:**
- Direct and actionable
- Assume intelligence, not knowledge
- Focus on user goals, not system features

**Structure:**
- Lead with the outcome
- Provide context as needed
- Use examples liberally
- Include troubleshooting

### Maintenance

**Each document includes:**
- Last updated date
- Version compatibility
- Prerequisites and assumptions
- Links to related content

## Implementation Status

### ✅ Completed (Tier 1 & 2)
- `README.md` - Updated with new structure
- `docs/index.rst` - Main documentation index
- `docs/quickstart.rst` - 5-minute getting started
- `docs/installation.rst` - Platform-specific installation
- `docs/cli-guide.rst` - Complete CLI usage
- `docs/simple-interface.rst` - Zero-config interface
- `docs/cli-reference.rst` - Complete command reference
- `docs/examples/index.rst` - Example organization

### 🚧 In Progress (Tier 3 & 4)
- `docs/python-api.rst` - Python API guide
- `docs/configuration.rst` - Configuration guide
- `docs/backends.rst` - Backend options
- `docs/api-reference.rst` - Complete API reference
- `docs/troubleshooting.rst` - Troubleshooting guide

### 📋 Planned (Tier 4 & 5)
- `docs/architecture.rst` - System architecture
- `docs/performance.rst` - Performance optimization
- Complete examples in `docs/examples/`
- Advanced topics documentation

## Building Documentation

### Local Development

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

### GitHub Pages Integration

The documentation is structured to work with:
- GitHub's automatic README rendering
- Sphinx for comprehensive documentation
- GitHub Pages for hosted documentation

## Success Metrics

**Documentation effectiveness measured by:**
- Time to first successful operation (target: <5 minutes)
- User retention after first use
- Reduction in support questions
- Community contribution rate

**User feedback channels:**
- GitHub Issues for documentation problems
- GitHub Discussions for usage questions
- Pull requests for documentation improvements

This tiered structure ensures that users can find the right level of information for their needs while maintaining comprehensive coverage of all SEM capabilities.
