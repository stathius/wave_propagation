from Lib_Refactor import *

for nr_net in range(20):
    """
    Naming of network for saving
    """
    version = nr_net + 10
    input_frames = 5
    output_frames = 10
    Type_Network = "7_kernel_3LSTM"
    DataGroup = "LSTM"


    # Little trick to adjust path files for compatibility (I have a backup of the Main.py in case it doesn't work)
    # stef_path = "/media/sg6513/DATADRIVE2/MSc/Wavebox/"
    # if os.path.isfile(stef_path + "stefpc.txt"):
    #     if not os.path.isdir(stef_path + "Results"):
    #         os.mkdir(stef_path + "Results")
    #     maindir1 = stef_path + "Results/Simulation_Result_" + Type_Network + "_v%03d/" % version
    #     maindir2 = stef_path
    #     version += 200
    # else:
    if not os.path.isdir("./Results"):
        os.mkdir("./Results")
    # maindir1 = "/mnt/Linux-HDD/Discrete_Data-sharing_LSTMs/Results/Simulation_Result_"\
    maindir1 = "./Results/Simulation_Result_"\
               + Type_Network + "_v%03d/" % version
    maindir2 = "./"
    if not os.path.isdir(maindir1):
        make_folder_results(maindir1)

    # Data
    if os.path.isfile(maindir1 + "All_Data_" + DataGroup + "_v%03d.pickle" % version):
        My_Data = load(maindir1 + "All_Data_" + DataGroup + "_v%03d" % version)
        My_Train = My_Data["Training data"]
        My_Validate = My_Data["Validation data"]
        My_Test = My_Data["Testing data"]
    else:
        My_Test, My_Validate, My_Train = Create_Training_Testing_Datasets(
            maindir2 + "Video_Data/", transformVar, test_fraction=0.15, validation_fraction=0.15, check_bad_data=False)
        My_Data = {"Training data": My_Train, "Validation data": My_Validate, "Testing data": My_Test}
        save(My_Data, maindir1 + "All_Data_" + DataGroup + "_v%03d" % version)

    # Analyser
    if os.path.isfile(maindir1 + Type_Network + "_Analyser_v%03d.pickle" % version):
        Analyser = load(maindir1 + Type_Network + "_Analyser_v%03d" % version)
    else:
        Analyser = Data_Analyser(maindir1, Type_Network, version)

    # Model
    if os.path.isfile(maindir1 + Type_Network + "_Project_v%03d.pt" % version):
        model = torch.load(maindir1 + Type_Network + "_Project_v%03d.pt" % version)
    else:
        model = Network()



    # Learning Rate scheduler w. optimizer
    if os.path.isfile(maindir1 + Type_Network + "_lrScheduler_v%03d.pickle" % version):
        scheduler_dict = load(maindir1 + Type_Network + "_lrScheduler_v%03d" % version)
        lrschedule = scheduler_dict["Type"]
        exp_lr_scheduler = scheduler_dict["Scheduler"]
    else:
        # Optimizer
        optimizer_algorithm = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        # Add learning rate schedulers
        # Decay LR by a factor of gamma every step_size epochs
        lrschedule = 'plateau'
        if lrschedule == 'step':
            gamma = 0.5
            step_size = 40
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_algorithm, step_size=step_size, gamma=gamma)
        elif lrschedule == 'plateau':
            # Reduce learning rate when a metric has stopped improving
            exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)
            optimizer_algorithm = []

    if torch.cuda.is_available():
        model.cuda()
    score_keeper = Scorekeeper()

    Train_Data = DataLoader(My_Train, batch_size=16, shuffle=True, num_workers=12)
    Validate_Data = DataLoader(My_Validate, batch_size=16, shuffle=True, num_workers=12)
    Test_Data = DataLoader(My_Test, batch_size=16, shuffle=True, num_workers=12)
    #Square_Data = DataLoader(My_Square, batch_size=16, shuffle=True, num_workers=12)
    #Line_Data = DataLoader(My_Line, batch_size=16, shuffle=True, num_workers=12)
    #Analyser.plot_loss()
    #Analyser.plot_accuracy()
    #Analyser.plot_loss_batchwise()
    Analyser.plot_validation_loss()

    """
    Main Code
    """
    for _ in range(50 - len(Analyser.epoch_loss)):
        print(version)
        for g in exp_lr_scheduler.optimizer.param_groups:
            print(g["lr"])
        """
        Here we can access Analyser.validation_loss to make decisions
        """
        # Learning rate scheduler
        # perform scheduler step if independent from validation loss
        if lrschedule == 'step':
            exp_lr_scheduler.step()
        train(len(Analyser.epoch_loss) + 1, Train_Data, Validate_Data, plot=False)
        # perform scheduler step if Dependent on validation loss
        if lrschedule == 'plateau':
            exp_lr_scheduler.step(Analyser.validation_loss[-1])
        save_network(model, maindir1 + Type_Network + "_Project_v%03d" % version)
        torch.save(model, maindir1 + Type_Network + "_Project_v%03d.pt" % version)
        save(Analyser, maindir1 + Type_Network + "_Analyser_v%03d" % version)
        scheduler_dict = {"Type": lrschedule, "Scheduler": exp_lr_scheduler}
        save(scheduler_dict, maindir1 + Type_Network + "_lrScheduler_v%03d" % version)
    test(Test_Data, plot=False)
    Analyser = []
    model =[]
    exp_lr_scheduler = []
    scheduler_dict = []
Analyser.plot_loss()
Analyser.plot_accuracy()
Analyser.plot_loss_batchwise()
Analyser.plot_validation_loss()