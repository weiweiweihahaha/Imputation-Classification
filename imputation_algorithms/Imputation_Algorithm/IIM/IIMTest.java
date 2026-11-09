package Imputation_Algorithm.IIM;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

/**
 * Created by Aoqian Zhang on 2019/6/28.
 * E-mail address is zaqthss2009@gmail.com
 * All Rights Reserved.
 *
 * @author Aoqian Zhang
 */

public class IIMTest {
   public static void writeCSV(double[][] array, String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (double[] row : array) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < row.length; j++) {
                    sb.append(row[j]);
                    if (j < row.length - 1) {
                        sb.append(",");
                    }
                }
                writer.write(sb.toString());
                writer.newLine();
            }
            //System.out.println("CSV file has been created successfully!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

  private static String buildLegacyOutputPath(String datasetName, String name) {
    String currentDirectory = System.getProperty("user.dir");
    File currentDirFile = new File(currentDirectory);
    String outputDir =
        currentDirFile
            + File.separator
            + "data"
            + File.separator
            + "Clean_data"
            + File.separator
            + name;
    return outputDir + File.separator + datasetName + "_IIM_clean.csv";
  }

  public static void main(String[] args) {
    String datasetName = args[0];
    String name = args[1];
    String inputPath = args.length > 2 ? args[2] : datasetName + ".csv";
    String outputPath = args.length > 3 ? args[3] : buildLegacyOutputPath(datasetName, name);
    int[] lparams = {1, 101, 5}; // LBEGIN, LMAX, INTERVAL
    int K = 10;

    // run IIM
    Database rawDb = DataUtil.readData(inputPath);
    IIM iim = new IIM(rawDb);
    iim.setParams(lparams, K);
    Map<Position, Cell> repairedCells = iim.mainIIM(true, false, true);
    try{
      FileReader fr = new FileReader(inputPath);
      FileReader fr_copy = new FileReader(inputPath);
      BufferedReader br = new BufferedReader(fr);
      BufferedReader br_copy = new BufferedReader(fr_copy);
      String line = "";
      int rows = 0;
      int cols = 0;
      String[] vals;
      while ((line = br_copy.readLine()) != null){
        rows = rows + 1;
        vals = line.split(",", -1);
        cols = vals.length;
      }
      double[][] array = new double[rows][cols];
      double[][] repaired = new double[rows][cols];
      for (Map.Entry<Position, Cell> entry : repairedCells.entrySet()) {  
        Position position = entry.getKey();  
        Cell cell = entry.getValue();   
        double cellValue = cell.getValue();
        repaired[position.gettIndex()][position.getAttrIndex()] = cellValue;
      } 
      int indexnum = 0;
      while ((line = br.readLine()) != null) {
          vals = line.split(",", -1);
          int attrNum = vals.length;
          for(int j = 0; j < attrNum; ++j){
            String value = vals[j].trim();
            if(value.isEmpty() || value.equalsIgnoreCase("nan")){
              array[indexnum][j] = repaired[indexnum][j];
            }else{
              array[indexnum][j] = Double.parseDouble(value);
            }
          }
          indexnum = indexnum + 1;
      }
      File outputFile = new File(outputPath);
      File outputDir = outputFile.getParentFile();
      if (outputDir != null) {
        outputDir.mkdirs();
      }
      writeCSV(array, outputPath);
      br.close();
      br_copy.close();
      fr.close();
      fr_copy.close();
    }catch (IOException e) {
      throw new RuntimeException("Failed to write imputed output: " + outputPath, e);
    }
    
  }
}
