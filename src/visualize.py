import pandas as pd

def sumary_issue(df):
    issue_columns = [col for col in df.columns if col.startswith('is_')]
    issue_counts = {}
    for column in issue_columns:
        issue_type = column.replace('is_', '')  
        num_issues = df[df[column]].shape[0]  
        issue_counts[issue_type] = num_issues
    report_df = pd.DataFrame(list(issue_counts.items()), columns=['issue_type', 'num_images'])
    report_df = report_df.sort_values(by='num_images', ascending=False).reset_index(drop=True)
    
    return report_df
