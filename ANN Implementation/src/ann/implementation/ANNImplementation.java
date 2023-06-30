package ann.implementation;

import java.io.*;
import java.util.*;
import static org.apache.poi.ss.usermodel.CellType.NUMERIC;
import org.apache.poi.ss.usermodel.FormulaEvaluator;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class ANNImplementation {

    private static final ArrayList<Sample> entireDataSet = new ArrayList<>();
    //Data subsets
    private static final ArrayList<Sample> testDataSet = new ArrayList<>(); //About 20% of data
    private static final ArrayList<Sample> validationDataSet = new ArrayList<>(); //About 20% of data
    private static final ArrayList<Sample> trainingDataSet = new ArrayList<>(); //About 60% of data

    private static double[] minimums;
    private static double[] maximums;

    private static final XSSFWorkbook workbook = new XSSFWorkbook();
    private static int dataPointsCount;

    //ANN parameters
    private static int[] ANN_SHAPE = new int[]{14, 12};
    private static double MOMENTUM = 0.9d;
    private static int MAX_EPOCHS = 10000;
    private static double START_STEP_SIZE = 0.1d;
    private static double END_STEP_SIZE = 0.01d;
    private static double MAX_STEP_SIZE = 0.5d;
    private static double MIN_STEP_SIZE = 0.001d;

    //ANN improvments
    private static boolean USE_MOMENTUM = true;
    private static boolean USE_BOLD_DRIVER = true;
    private static boolean USE_ANNEALING = false;
    private static boolean USE_WEIGHT_DECAY = false;
    private static boolean USE_BATCH_LEARING = false;

    private static final Random rand = new Random();

    public static void main(String[] args) throws IOException {

        Scanner input = new Scanner(System.in);

        //Load model paramaters from config.txt
        loadCongfig();
        //Read and process data set from DataSet.xlsx - test it option 3
        processDataSet();

        boolean askAgain = true;
        //Give the user a list of options
        while (askAgain) {
            try {
                System.out.println("\nWould you like to:\n"
                        + "(1) Train an new ANN on labelled data set?\n"
                        + "(2) Evaluate an existing ANN on labelled data set?\n"
                        + "(3) Execute existing ANN on unlabelled data set?\n"
                        + "(4) Save output and quit?");
                int choice = input.nextInt();
                ArrayList<Neuron[]> neuronLayers = null;
                switch (choice) {
                    //If Train an new ANN
                    case (1):
                        //Train MLP with given parameters
                        neuronLayers = trainANN();
                        System.out.println("\nModel Trained.");
                        boolean success = false;
                        while (!success) {
                            //Attempt to load text file specifed by user
                            System.out.println("Enter file name (not file path), enter \"no\" to not save ANN: ");
                            String fileName = input.next();
                            if (fileName.equals("no") || fileName.equals("NO") || fileName.equals("No")) {
                                break;
                            }
                            success = saveANN(neuronLayers, fileName);
                            if (success) {
                                System.out.println("Model Saved.");
                            }
                        }
                        break;

                    //If Evaluate an existing ANN
                    case (2):

                        //Continue to ask until valid file chosen
                        while (neuronLayers == null) {
                            //Attempt to load text file specifed by user
                            System.out.println("\nEnter file name (not file path), enter \"no\" to cancel: ");
                            String fileName = input.next();
                            if (fileName.equals("no") || fileName.equals("NO") || fileName.equals("No")) {
                                break;
                            }
                            neuronLayers = loadANN(fileName);
                            //If file exist with correct format
                            if (neuronLayers != null) {
                                //If ANN compatible with data set
                                if (neuronLayers.get(0)[0].getWeights().length == dataPointsCount) {
                                    //Evaluates data set using test set
                                    double[] evaluationMetrics = evaluateANN(neuronLayers, testDataSet, true);
                                    System.out.println("Evaluation Metrics: ");
                                    System.out.printf("\nRMSE – Root Mean Squared Error: \t\t\t%.4f \t(closer to 0 is better)\n", evaluationMetrics[0]);
                                    System.out.printf("SRE - Mean Squared Relative Error: \t\t%.4f \t(closer to 0 is better)\n", evaluationMetrics[1]);
                                    System.out.printf("CE - Coefficient of Efficiency: \t\t%.4f \t(closer to 1 is better)\n", evaluationMetrics[2]);
                                    System.out.printf("RSQR - R-Squared (Determination Coefficien): \t%.4f \t(closer to 1 is better)\n", evaluationMetrics[3]);
                                } else {
                                    neuronLayers = null;
                                    System.out.println("ANN not compatible with data set - incorrect number of inputs");
                                }
                            } else {
                                System.out.println("Incorrect file name");
                            }
                        }
                        break;

                    //If Execute existing ANN on entire data set
                    case (3):
                        //Continue to ask until valid file chosen
                        while (neuronLayers == null) {
                            //Attempt to load text file specifed by user
                            System.out.println("\nEnter file name (not file path), enter \"no\" to cancel: ");
                            String fileName = input.next();
                            if (fileName.equals("no") || fileName.equals("NO") || fileName.equals("No")) {
                                break;
                            }
                            neuronLayers = loadANN(fileName);
                            //If file exist with correct format
                            if (neuronLayers != null) {
                                //If ANN compatible with data set
                                if (neuronLayers.get(0)[0].getWeights().length == dataPointsCount + 1) {
                                    //Executes data set on all inputs in data set
                                    executeANN(neuronLayers, entireDataSet);
                                    System.out.println("Model Executed.");
                                } else {
                                    neuronLayers = null;
                                    System.out.println("ANN not compatible with data set - incorrect number of inputs");
                                }
                            } else {
                                System.out.println("Incorrect file name");
                            }
                        }
                        break;

                    //If Save output and quit
                    case (4):
                        askAgain = false;
                        break;

                    default:
                        System.out.println("Incorrect choice, try again");
                        askAgain = true;
                        break;
                }
            } catch (Exception e) {
                System.out.println("Incorrect choice, try again");
                input.next();
            }
        }

        //Save output to "Output.xlsx"
        try {
            FileOutputStream out = new FileOutputStream(new File("Output.xlsx"));
            workbook.write(out);
            out.close();
            System.out.println("\nSuccessfully saved output file");
        } catch (IOException e) {
            System.out.println("\nError creating output file. Error: " + e);
        }

    }

    private static void loadCongfig() {
        String dir = System.getProperty("user.dir") + "\\config.txt";
        File file = new File(dir);
        boolean defaultValues = false;

        if (file.exists()) {
            //Try get every value from config file
            try {
                Scanner reader = new Scanner(file);
                //Read each line in text file
                while (reader.hasNextLine()) {
                    //Convert each line to kay, value pair
                    String[] stringData = reader.nextLine().split("=");
                    if (stringData.length != 2) {
                        System.out.println("nahh");
                        continue;
                    }
                    String key = stringData[0].trim();
                    String value = stringData[1].trim();
                    //Set value for specified paramater
                    switch (key) {
                        case ("ANN_SHAPE"):
                            if (value.equals("")) {
                                ANN_SHAPE = new int[0];
                            } else {
                                String[] layerSizes = value.split(",");
                                ANN_SHAPE = new int[layerSizes.length];
                                for (int i = 0; i < layerSizes.length; i++) {
                                    ANN_SHAPE[i] = Integer.valueOf(layerSizes[i].trim());
                                }
                            }
                            break;
                        case ("MOMENTUM"):
                            MOMENTUM = Double.valueOf(value);
                            break;
                        case ("MAX_EPOCHS"):
                            MAX_EPOCHS = Integer.valueOf(value);
                            break;
                        case ("START_STEP_SIZE"):
                            START_STEP_SIZE = Double.valueOf(value);
                            break;
                        case ("END_STEP_SIZE"):
                            END_STEP_SIZE = Double.valueOf(value);
                            break;
                        case ("MAX_STEP_SIZE"):
                            MAX_STEP_SIZE = Double.valueOf(value);
                            break;
                        case ("MIN_STEP_SIZE"):
                            MIN_STEP_SIZE = Double.valueOf(value);
                            break;
                        case ("USE_MOMENTUM"):
                            USE_MOMENTUM = Boolean.valueOf(value);
                            break;
                        case ("USE_BOLD_DRIVER"):
                            USE_BOLD_DRIVER = Boolean.valueOf(value);
                            break;
                        case ("USE_ANNEALING"):
                            USE_ANNEALING = Boolean.valueOf(value);
                            break;
                        case ("USE_WEIGHT_DECAY"):
                            USE_WEIGHT_DECAY = Boolean.valueOf(value);
                            break;
                        case ("USE_BATCH_LEARING"):
                            USE_BATCH_LEARING = Boolean.valueOf(value);
                            break;
                    }
                }
                reader.close();
            } //If there is any errors (because of incorrect format) then use default values instead
            catch (Exception e) {
                System.out.println("An error occurred loading config file. Error: " + e);
                defaultValues = true;
            }
        } //If there is no config file then use default values instead
        else {
            System.out.println("Config file does not exist, using default values.");
            defaultValues = true;
        }
        //Reset model paramaters to default if necessary
        if (defaultValues) {
            ANN_SHAPE = new int[]{10, 8};
            MOMENTUM = 0.9;
            MAX_EPOCHS = 1000;
            START_STEP_SIZE = 0.1;
            END_STEP_SIZE = 0.01;
            MAX_STEP_SIZE = 0.5;
            MIN_STEP_SIZE = 0.001;
            USE_MOMENTUM = true;
            USE_BOLD_DRIVER = true;
            USE_ANNEALING = false;
            USE_WEIGHT_DECAY = false;
            USE_BATCH_LEARING = false;
        }
        //Display used model paramaters
        System.out.println("ANN parameters:");
        System.out.println("ANN_SHAPE = " + Arrays.toString(ANN_SHAPE));
        System.out.println("MOMENTUM = " + MOMENTUM);
        System.out.println("MAX_EPOCHS = " + MAX_EPOCHS);
        System.out.println("START_STEP_SIZE = " + START_STEP_SIZE);
        System.out.println("END_STEP_SIZE = " + END_STEP_SIZE);
        System.out.println("MAX_STEP_SIZE = " + MAX_STEP_SIZE);
        System.out.println("MIN_STEP_SIZE = " + MIN_STEP_SIZE);
        System.out.println("USE_MOMENTUM = " + USE_MOMENTUM);
        System.out.println("USE_BOLD_DRIVER = " + USE_BOLD_DRIVER);
        System.out.println("USE_ANNEALING = " + USE_ANNEALING);
        System.out.println("USE_WEIGHT_DECAY = " + USE_WEIGHT_DECAY);
        System.out.println("USE_BATCH_LEARING = " + USE_BATCH_LEARING);

        if (!USE_MOMENTUM) {
            MOMENTUM = 0;
        }
    }

    private static void processDataSet() throws FileNotFoundException, IOException {

        final ArrayList<Sample> dataSet = new ArrayList<>();

        //Read data set from DataSet.xlsx
        FileInputStream inputFile = new FileInputStream(new File("DataSet.xlsx"));
        XSSFWorkbook dataSetWorkbook = new XSSFWorkbook(inputFile);
        XSSFSheet sheet = dataSetWorkbook.getSheetAt(0);
        FormulaEvaluator evaluator = dataSetWorkbook.getCreationHelper().createFormulaEvaluator();

        int sampleCount = sheet.getPhysicalNumberOfRows();
        dataPointsCount = sheet.getRow(0).getPhysicalNumberOfCells();

        //Extra information for each column of data (used for data pre-processing)
        minimums = new double[dataPointsCount];
        maximums = new double[dataPointsCount];

        //Read every row in sheet
        for (int row = 0; row < sampleCount; row++) {
            //Set to false if a non-numeric value is detected in a sample
            boolean validSample = true;
            //Read row from data set
            double[] sampleData = new double[dataPointsCount];
            for (int col = 0; col < dataPointsCount; col++) {
                if (sheet.getRow(row).getCell(col) != null && evaluator.evaluateInCell(
                        sheet.getRow(row).getCell(col)).getCellTypeEnum() == NUMERIC) {
                    sampleData[col] = sheet.getRow(row).getCell(col).getNumericCellValue();
                    //Calculate min and max values for input (column), used for standardization later
                    minimums[col] = Math.min(minimums[col], sampleData[col]);
                    maximums[col] = Math.max(maximums[col], sampleData[col]);
                } else {
                    validSample = false;
                    break;
                }
            }
            //Invalid samples are ignored
            if (!validSample) {
                continue;
            }
            //Convert each row of data into a Sample data object and add it to dataSet
            dataSet.add(new Sample(sampleData));
        }

        System.out.println("\nReading Data Set:");

        //Loop through data set
        for (Sample sample : dataSet) {
            double[] sampleData = sample.getData();
            //Standardize each value between range [0.1, 0.9]
            for (int i = 0; i < sampleData.length; i++) {
                sampleData[i] = standardize(sampleData[i], maximums[i], minimums[i]);
            }
            //Display sample data
            System.out.println(sample);
            //Randomly place sample between 3 data sets
            switch (rand.nextInt(5)) {
                case (0): //20% chance
                    testDataSet.add(sample);
                    break;
                case (1): //20% chance
                    validationDataSet.add(sample);
                    break;
                default: //60% chance
                    trainingDataSet.add(sample);
                    break;
            }
            entireDataSet.add(sample);
        }
        //Display data subset sizes
        System.out.printf("\nTotal number of samples: %d\n", entireDataSet.size());
        System.out.printf("Test set size: %.2f%%\n", (double) testDataSet.size() / dataSet.size() * 100);
        System.out.printf("Validation set size: %.2f%%\n", (double) validationDataSet.size() / dataSet.size() * 100);
        System.out.printf("Training set size: %.2f%%\n", (double) trainingDataSet.size() / dataSet.size() * 100);
    }

    private static double standardize(double value, double max, double min) {
        //Standardizes value between range [0.1, 0.9]
        return 0.8 * ((value - min) / (max - min)) + 0.1;
    }

    private static double deStandardize(double value, double max, double min) {
        //Converts value back from range [0.1, 0.9] to normal range
        return ((value - 0.1) / 0.8) * (max - min) + min;
    }

    private static ArrayList<Neuron[]> trainANN() {

        if (workbook.getSheetIndex("Training Data") != -1) {
            workbook.removeSheetAt(workbook.getSheetIndex("Training Data"));
        }
        XSSFSheet spreadsheet = workbook.createSheet("Training Data");
        //Save epoch data Headers
        XSSFRow header = spreadsheet.createRow(0);
        header.createCell(0).setCellValue("Epoch");
        header.createCell(1).setCellValue("RMSE");
        header.createCell(2).setCellValue("Step size");
        int line = 1;

        double stepSize = START_STEP_SIZE;

        //Array list of neuron layers represents ANN
        final ArrayList<Neuron[]> neuronLayers = new ArrayList<>();
        //Create uninitialised hidden layers of neurons
        for (int hiddenLayerSize : ANN_SHAPE) {
            if (hiddenLayerSize > 0) {
                neuronLayers.add(new Neuron[hiddenLayerSize]);
            }
        }
        //Create uninitialised output neuron
        neuronLayers.add(new Neuron[1]);
        //Initialise neurons with random weights
        initialiseNeuronsWithRandomWeights(neuronLayers);

        //Copy of previous neural network instance (used for bold driver)
        final ArrayList<Neuron[]> prevNeuronLayers = new ArrayList<>();
        for (int hiddenLayerSize : ANN_SHAPE) {
            if (hiddenLayerSize > 0) {
                prevNeuronLayers.add(new Neuron[hiddenLayerSize]);
            }
        }
        prevNeuronLayers.add(new Neuron[1]);
        cloneNeuronLayers(neuronLayers, prevNeuronLayers);

        //Previously measured RMSE
        double error = Double.MAX_VALUE;
        int tinyErrorCount = 0;

        //Train for specified number of epochs
        for (int i = 0; i < MAX_EPOCHS; i++) {
            //Loop through every sample in training set
            for (Sample sample : trainingDataSet) {
                //Inputs and bias passed to first layer of hidden neurons
                double[] inputsAndBias = new double[sample.getData().length];
                System.arraycopy(sample.getData(), 0, inputsAndBias, 0, inputsAndBias.length - 1);
                inputsAndBias[inputsAndBias.length - 1] = 1;
                //Correct output specified by sample
                double sampleOutput = sample.getData()[sample.getData().length - 1];
                //Forward pass through ANN
                forwardPass(neuronLayers, inputsAndBias);
                //Backward pass through ANN
                backwardPass(neuronLayers, sampleOutput, stepSize, i + 1);

                if (!USE_BATCH_LEARING) {
                    //Update weights for every neuron in ANN
                    updateWeights(neuronLayers, stepSize);
                } else {
                    //Sum error gradient for all weights
                    sumWeightsErrorGradient(neuronLayers);
                }
            }

            //Update weights at the end of the epoch
            if (USE_BATCH_LEARING) {
                batchUpdateWeights(neuronLayers, stepSize, trainingDataSet.size());
            }

            //Simulated annealing
            if (USE_ANNEALING) {
                stepSize = annealing(i + 1);
            }

            //100 times while training (every 1% of training complete)
            if (i % (Math.max(MAX_EPOCHS / 100, 1)) == 0) {
                //Get Root Mean Squared Error from evaluating model on validation set
                double RMSE = evaluateANN(neuronLayers, validationDataSet, false)[0];

                if (USE_BOLD_DRIVER) {
                    double errorDiff = ((RMSE - error) / error);
                    //If the error increases by over 1 % then half the step size and revert the weights back to the prev weights
                    if (errorDiff > 0.01d) {
                        stepSize *= 0.5;
                        stepSize = Math.max(MIN_STEP_SIZE, stepSize);
                        //revert model to last bold driver
                        cloneNeuronLayers(prevNeuronLayers, neuronLayers);
                    } else {
                        //If the error decreases by over 1 % then slightly increase step size
                        if (errorDiff < -0.01d) {
                            stepSize *= 1.05;
                            stepSize = Math.min(MAX_STEP_SIZE, stepSize);
                        }
                        cloneNeuronLayers(neuronLayers, prevNeuronLayers);
                    }
                }
                //Display epoch data
                System.out.printf("\nEpoch %d completed - %.2f%% \tStep Size:%f \tError:%f", i, (double) i / MAX_EPOCHS * 100, stepSize, RMSE);

                //Save epoch data
                XSSFRow row = spreadsheet.createRow(line++);
                row.createCell(0).setCellValue(i);
                row.createCell(1).setCellValue(RMSE);
                row.createCell(2).setCellValue(stepSize);

                //Terminate training early if error change is negligible 3 times in a row
                if (Math.abs(RMSE - error) < 0.00001d) {
                    tinyErrorCount++;
                    if (tinyErrorCount >= 3) {
                        System.out.println(" - Change in Error minimal, stopping training");
                        break;
                    }
                } else {
                    tinyErrorCount = 0;
                }
                error = RMSE;
            }
        }

        return neuronLayers;
    }

    private static void initialiseNeuronsWithRandomWeights(ArrayList<Neuron[]> neuronLayers) {
        //Initialise all neurons with random weights and bias
        for (int i = 0; i < neuronLayers.size(); i++) {
            int weightsCount;
            //If first layer of hidden neurons then #weights = #sampleInputs + 1
            if (i == 0) {
                weightsCount = dataPointsCount;
            } //Otherwise, #weights = #previousLayerNeurons + 1
            else {
                weightsCount = neuronLayers.get(i - 1).length + 1;
            }
            for (int j = 0; j < neuronLayers.get(i).length; j++) {
                //Number of weights (#inputs + 1 for bias) for neuron
                double[] weightsAndBias = new double[weightsCount];
                //Randomise weights and bias values between [-2/n, 2/n] where n is the input size of the neuron
                int range = weightsAndBias.length - 1;
                for (int k = 0; k < weightsAndBias.length; k++) {
                    //Random value between [-2/n, 2/n] where n is the input size
                    weightsAndBias[k] = -2d / range + (2d / range - -2d / range) * rand.nextDouble();
                }
                neuronLayers.get(i)[j] = new Neuron(weightsAndBias);
            }
        }
    }

    private static void cloneNeuronLayers(ArrayList<Neuron[]> original, ArrayList<Neuron[]> clone) {
        //Copy weights values from original to clone
        for (int i = 0; i < original.size(); i++) {
            for (int j = 0; j < original.get(i).length; j++) {
                clone.get(i)[j] = new Neuron(original.get(i)[j]);
            }
        }
    }

    private static double forwardPass(ArrayList<Neuron[]> neuronLayers, double[] sampleInputsAndBias) {
        //Outputs from every neuron in the layer, used as inputs to the next layer
        double[] layerOutputs = new double[0];

        //Forward pass through every layer in ANN
        for (int i = 0; i < neuronLayers.size(); i++) {
            //Inputs and bias for every neuron in the layer
            double[] inputsAndBias;
            //If first layer then inputs equal to inputs from the sample
            if (i == 0) {
                inputsAndBias = sampleInputsAndBias;
            } //Otherwise, inputs equal to outputs of previous layer
            else {
                inputsAndBias = layerOutputs;
            }
            //Clears layer outputs and sets bias input to 1
            layerOutputs = new double[neuronLayers.get(i).length + 1];
            layerOutputs[layerOutputs.length - 1] = 1;
            //Set inputs, calculate weight sum, compute activation for every neuron in layer
            for (int j = 0; j < neuronLayers.get(i).length; j++) {
                neuronLayers.get(i)[j].setInputs(inputsAndBias);
                neuronLayers.get(i)[j].computeWeightSum();
                layerOutputs[j] = neuronLayers.get(i)[j].computeActivation();
            }
        }
        //Returns the output of the final neuron (output neuron), the output of the entire ANN
        return layerOutputs[layerOutputs.length - 2];
    }

    private static void backwardPass(ArrayList<Neuron[]> neuronLayers, double sampleOutput, double stepSize, int epoch) {
        //Backward pass through every layer in ANN
        for (int i = neuronLayers.size() - 1; i >= 0; i--) {
            //If last layer (output layers)
            if (i == neuronLayers.size() - 1) {
                //Compute delta value for output neuron
                for (Neuron outputNeuron : neuronLayers.get(i)) {
                    // Cacuiate upsilon (for weight decay)
                    double upsilon = 1 / (stepSize * epoch);
                    if (!USE_WEIGHT_DECAY || epoch < 100) {
                        upsilon = 0;
                    }
                    //Compute delta value for output neuron
                    outputNeuron.computeDeltaValue(sampleOutput, upsilon);
                }
            } //Otherwise, if hidden layer
            else {
                //Loop through neurons in hidden layer
                for (int j = 0; j < neuronLayers.get(i).length; j++) {
                    double[] forwardWeights = new double[neuronLayers.get(i + 1).length];
                    double[] forwardDeltaValues = new double[neuronLayers.get(i + 1).length];
                    //Loop through forward neurons
                    for (int k = 0; k < neuronLayers.get(i + 1).length; k++) {
                        Neuron forwardNeuron = neuronLayers.get(i + 1)[k];
                        //Store weights between current neuron and all forward neurons
                        forwardWeights[k] = forwardNeuron.getWeights()[j];
                        //Store delta values of every forward neuron
                        forwardDeltaValues[k] = forwardNeuron.getDeltaValue();
                    }
                    //Compute delta value for hidden neuron
                    neuronLayers.get(i)[j].computeDeltaValue(forwardWeights, forwardDeltaValues);
                }
            }
        }
    }

    private static void updateWeights(ArrayList<Neuron[]> neuronLayers, double stepSize) {
        //Calculates new weights for every neuron in ANN
        for (Neuron[] neuronLayer : neuronLayers) {
            for (Neuron neuron : neuronLayer) {
                neuron.updateWeightsAndBias(stepSize, MOMENTUM);
            }
        }
    }

    private static void batchUpdateWeights(ArrayList<Neuron[]> neuronLayers, double stepSize, int sampleCount) {
        //Calculates new weights for every neuron in ANN
        for (Neuron[] neuronLayer : neuronLayers) {
            for (Neuron neuron : neuronLayer) {
                neuron.batchUpdateWeightAndBias(stepSize, MOMENTUM, sampleCount);
            }
        }
    }

    private static void sumWeightsErrorGradient(ArrayList<Neuron[]> neuronLayers) {
        //sum every weights error gradient for every neuron in ANN
        for (Neuron[] neuronLayer : neuronLayers) {
            for (Neuron neuron : neuronLayer) {
                neuron.sumWeightsErrorGradient();
            }
        }
    }

    private static double annealing(double epoch) {
        //Decays stepSize (learning parameter) over time
        return END_STEP_SIZE + (START_STEP_SIZE - END_STEP_SIZE)
                * (1 - (1 / (1 + Math.pow(Math.E, (10 - 20 * (epoch / MAX_EPOCHS))))));
    }

    private static double[] evaluateANN(ArrayList<Neuron[]> neuronLayers, ArrayList<Sample> dataSubset, boolean store) {
        XSSFSheet spreadsheet = null;
        int line = 1;
        if (store) {
            if (workbook.getSheetIndex("Evaluation Data") != -1) {
                workbook.removeSheetAt(workbook.getSheetIndex("Evaluation Data"));
            }
            spreadsheet = workbook.createSheet("Evaluation Data");
            //Save evaluation data Headers
            XSSFRow header = spreadsheet.createRow(0);
            header.createCell(0).setCellValue("Sample outputs");
            header.createCell(1).setCellValue("Modelled output");
        }

        //RMSE, MSRE, CE, RSQR
        double[] evaluationMetrics = new double[4];

        double[] sampleOutputs = new double[dataSubset.size()];
        double[] modelledOutputs = new double[dataSubset.size()];

        //Variables used for calculation
        double RMSEsum = 0, MSREsum = 0, CEnumerator = 0, CEdenominator = 0, RSQRnumerator = 0, RSQRdenominatorLeft = 0, RSQRdenominatorRight = 0;

        double sampleOutputsMean = 0;
        double modelledOutputsMean = 0;
        //Store sample output and modelled output for every sample in given data subset
        for (int i = 0; i < dataSubset.size(); i++) {
            Sample sample = dataSubset.get(i);
            //Inputs and bias passed to first layer of hidden neurons
            double[] inputsAndBias = new double[sample.getData().length];
            System.arraycopy(sample.getData(), 0, inputsAndBias, 0, inputsAndBias.length - 1);
            inputsAndBias[inputsAndBias.length - 1] = 1;

            //Correct output specified by sample
            double sampleOutput = sample.getData()[sample.getData().length - 1];
            //De-standardize sample output
            sampleOutputs[i] = deStandardize(sampleOutput, maximums[maximums.length - 1], minimums[minimums.length - 1]);
            //Forward pass through ANN
            double modelledOutput = forwardPass(neuronLayers, inputsAndBias);
            //De-standardize modelled output
            modelledOutputs[i] = deStandardize(modelledOutput, maximums[maximums.length - 1], minimums[minimums.length - 1]);

            //Calculate mean for both values
            sampleOutputsMean += sampleOutputs[i];
            modelledOutputsMean += modelledOutputs[i];

            //Save evaluation data 
            if (store && spreadsheet != null) {
                XSSFRow row = spreadsheet.createRow(line++);
                row.createCell(0).setCellValue(sampleOutputs[i]);
                row.createCell(1).setCellValue(modelledOutputs[i]);
            }
        }

        sampleOutputsMean /= dataSubset.size();
        modelledOutputsMean /= dataSubset.size();

        //Calculate evaluation metrics
        for (int i = 0; i < dataSubset.size(); i++) {
            RMSEsum += Math.pow(modelledOutputs[i] - sampleOutputs[i], 2);

            MSREsum += Math.pow((modelledOutputs[i] - sampleOutputs[i]) / sampleOutputs[i], 2);

            CEnumerator += Math.pow(modelledOutputs[i] - sampleOutputs[i], 2);
            CEdenominator += Math.pow(sampleOutputs[i] - sampleOutputsMean, 2);

            RSQRnumerator += (sampleOutputs[i] - sampleOutputsMean) * (modelledOutputs[i] - modelledOutputsMean);
            RSQRdenominatorLeft += Math.pow(sampleOutputs[i] - sampleOutputsMean, 2);
            RSQRdenominatorRight += Math.pow(modelledOutputs[i] - modelledOutputsMean, 2);
        }
        //RMSE – Root Mean Squared Error (closer to 0 is better)
        evaluationMetrics[0] = Math.sqrt(RMSEsum / dataSubset.size());
        //MSRE - Mean Squared Relative Error (closer to 0 is better)
        evaluationMetrics[1] = MSREsum / dataSubset.size();
        //CE - Coefficient of Efficiency (closer to 1 is better)
        evaluationMetrics[2] = 1 - CEnumerator / CEdenominator;
        //RSQR - R-Squared (Coefficient of Determination) (closer to 1 is better)
        evaluationMetrics[3] = Math.pow(RSQRnumerator / (Math.sqrt(RSQRdenominatorLeft * RSQRdenominatorRight)), 2);

        return evaluationMetrics;
    }

    private static void executeANN(ArrayList<Neuron[]> neuronLayers, ArrayList<Sample> dataSubset) {
        int line = 1;
        if (workbook.getSheetIndex("Execution Data") != -1) {
            workbook.removeSheetAt(workbook.getSheetIndex("Execution Data"));
        }
        XSSFSheet spreadsheet = workbook.createSheet("Execution Data");
        //Save evaluation data Headers
        XSSFRow header = spreadsheet.createRow(0);
        header.createCell(0).setCellValue("Modelled output (standardized)");

        //Store modelled output for every sample in given data subset
        for (int i = 0; i < dataSubset.size(); i++) {
            Sample sample = dataSubset.get(i);
            //Inputs and bias passed to first layer of hidden neurons
            double[] inputsAndBias;
            inputsAndBias = new double[sample.getData().length + 1];
            System.arraycopy(sample.getData(), 0, inputsAndBias, 0, inputsAndBias.length - 1);
            inputsAndBias[inputsAndBias.length - 1] = 1;

            //Forward pass through ANN
            double modelledOutput = forwardPass(neuronLayers, inputsAndBias);

            //Save execution data 
            XSSFRow row = spreadsheet.createRow(line++);
            row.createCell(0).setCellValue(modelledOutput);
        }
    }

    private static boolean saveANN(ArrayList<Neuron[]> neuronLayers, String fileName) {
        String dir = System.getProperty("user.dir") + "\\ANN Models\\" + fileName;

        try {
            File file = new File(dir);
            //Overwrites file if it already exits
            if (!file.createNewFile()) {
                file.delete();
            }

            FileWriter writeToFile = new FileWriter(dir, true);
            PrintWriter printToFile = new PrintWriter(writeToFile);

            String ANNweights = "";
            for (Neuron[] neuronLayer : neuronLayers) {
                //Write ANN structure to fist line
                printToFile.print(neuronLayer.length + ",");
                //Write weights for each neuron on seperate lines
                for (Neuron neuron : neuronLayer) {
                    String neuronWeights = "";
                    for (double weight : neuron.getWeights()) {
                        neuronWeights += weight + ",";
                    }
                    ANNweights += "\n" + neuronWeights;
                }
            }
            printToFile.println(ANNweights);
            printToFile.close();
            writeToFile.close();
        } catch (IOException e) {
            System.out.println("An error occurred saving ANN. Error: " + e);
            return false;
        }
        return true;
    }

    private static ArrayList<Neuron[]> loadANN(String fileName) {

        ArrayList<double[]> fileNumericData = new ArrayList<>();

        String dir = System.getProperty("user.dir") + "\\ANN Models\\" + fileName;

        File file = new File(dir);
        if (file.exists()) {
            try {
                Scanner reader = new Scanner(file);
                //Read each line in text file
                while (reader.hasNextLine()) {
                    //Convert each line to array of doubles
                    String[] stringData = reader.nextLine().split(",");
                    double[] numericData = new double[stringData.length];
                    for (int i = 0; i < stringData.length; i++) {
                        numericData[i] = Double.valueOf(stringData[i]);
                    }
                    fileNumericData.add(numericData);
                }
                reader.close();
                //Get ANN structure (number of layers and neurons per layer from first line)
                int[] hiddenLayerSizes = new int[fileNumericData.get(0).length];
                for (int i = 0; i < fileNumericData.get(0).length; i++) {
                    hiddenLayerSizes[i] = (int) fileNumericData.get(0)[i];
                }
                //Create array list of neuron layers represents ANN
                final ArrayList<Neuron[]> neuronLayers = new ArrayList<>();
                //Create uninitialised hidden layers of neurons
                for (int hiddenLayerSize : hiddenLayerSizes) {
                    neuronLayers.add(new Neuron[hiddenLayerSize]);
                }
                int line = 1;
                //Initialise all neurons with weights from the file
                for (int i = 0; i < neuronLayers.size(); i++) {
                    for (int j = 0; j < neuronLayers.get(i).length; j++) {
                        neuronLayers.get(i)[j] = new Neuron(fileNumericData.get(line++));
                    }
                }
                //Return loaded ANN
                return neuronLayers;
            } catch (IOException e) {
                System.out.println("An error occurred loading ANN. Error: " + e);
            }
        } else {
            System.out.println("The file does not exist.");
        }

        return null;
    }
}
