def patch_ultralytics():
    """Patch ultralytics to handle signals only in main thread"""
    try:
        import ultralytics
        from pathlib import Path
        
        # Get the path to the session.py file
        # Update this path based on your actual file location
        session_path = Path(ultralytics.__path__[0]) / 'hub/session.py'
        
        if not session_path.exists():
            print("Could not find ultralytics session.py file")
            return
        
        # Read the current content
        with open(session_path, 'r') as f:
            content = f.read()
        
        # Add thread check before signal handling
        if 'signal.signal(signal.SIGTERM, signal_handler)' in content:
            modified_content = content.replace(
                'signal.signal(signal.SIGTERM, signal_handler)',
                '''if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, signal_handler)'''
            )
            
            # Write the modified content back
            with open(session_path, 'w') as f:
                f.write(modified_content)
                
            print("Successfully patched ultralytics signal handling")
            
    except Exception as e:
        print(f"Failed to patch ultralytics: {str(e)}")
