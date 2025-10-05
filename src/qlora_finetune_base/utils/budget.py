def manage_budget(total_budget, used_budget):
    if used_budget > total_budget:
        raise ValueError("Used budget exceeds total budget.")
    remaining_budget = total_budget - used_budget
    return remaining_budget

def log_budget_usage(total_budget, used_budget):
    remaining_budget = manage_budget(total_budget, used_budget)
    print(f"Total Budget: {total_budget}, Used Budget: {used_budget}, Remaining Budget: {remaining_budget}")

def allocate_budget(total_budget, allocation_percentage):
    allocation = total_budget * allocation_percentage
    if allocation > total_budget:
        raise ValueError("Allocation exceeds total budget.")
    return allocation

def reset_budget():
    return 0