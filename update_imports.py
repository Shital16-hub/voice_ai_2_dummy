# update_imports.py
"""
Script to update import references from advanced_monitoring_system to main
"""
import os
import re

def update_file_imports(filename):
    """Update imports in a single file"""
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping...")
        return
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace import statements
        patterns = [
            (r'from advanced_monitoring_system import', 'from main import'),
            (r'import advanced_monitoring_system', 'import main'),
            (r'advanced_monitoring_system\.', 'main.'),
        ]
        
        updated = False
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                updated = True
        
        if updated:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated imports in {filename}")
        else:
            print(f"ℹ️  No imports to update in {filename}")
            
    except Exception as e:
        print(f"❌ Error updating {filename}: {e}")

def main():
    """Update all files that might reference advanced_monitoring_system"""
    files_to_update = [
        'test_monitoring_system.py',
        'demo_transcript_analysis.py',
        'enhanced_config.py',
        'quick_test.py',
    ]
    
    print("🔄 Updating import references...")
    
    for filename in files_to_update:
        update_file_imports(filename)
    
    print("\n✅ Import updates complete!")
    print("\n📋 Next steps:")
    print("1. Rename: advanced_monitoring_system.py → main.py")
    print("2. Test: python test_monitoring_system.py")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main()