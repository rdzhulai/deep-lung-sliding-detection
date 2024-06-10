def print_history(history, mode='train', epoch=0):
    mode_history = history.get(mode.lower(), None)
    if mode_history is None:
        print(f"Error: Mode '{mode}' not found in history.")
        return
    
    loss = mode_history['loss'][epoch]
    accuracy = mode_history['accuracy'][epoch]
    precision = mode_history['precision'][epoch]
    recall = mode_history['recall'][epoch]
    f1 = mode_history['f1'][epoch]
    spec = mode_history['specificity'][epoch]
    cm = mode_history['confusion_matrix'][epoch]
    time = mode_history['time'][epoch]
    
    print(f"{mode.capitalize()} - loss: {loss:.8f}, acc: {accuracy:.4f}, prec: {precision:.4f}, rec: {recall:.4f}, f1: {f1:.4f}, spec: {spec: .4f}, TP={cm[0][0]}, TN={cm[1][1]}, FP={cm[0][1]}, FN={cm[1][0]}, time: {time:.2f}s")