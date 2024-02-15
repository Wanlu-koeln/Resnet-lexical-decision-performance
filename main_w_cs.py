from ml_functions.resnet_add import train_and_test_sequentially


if __name__ == '__main__':
    N = 7650
    folder_list = ["original_oPE", "50p_oPE","normal"]  # Adjust this list as needed
    N_train_samples_list = [round(N * 0.01),round(N * 0.02),round(N * 0.03), round(N * 0.04),round(N * 0.05),
                            round(N * 0.1),round(N * 0.2),round(N * 0.3),round(N * 0.4),round(N * 0.5),round(N * 1)]  # Specify your sample sizes or generate programmatically
    base_folder = '/Users/wanlufu/Downloads/Dataset_oPE_24052023-4/'  # Change this to your data folder path
    task = "w_cs"  # Specify your task
    iterations = 50 # Number of iterations you want

    train_and_test_sequentially(folder_list, N_train_samples_list, base_folder, task, iterations)
