package Imputation_Algorithm.DLM;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
public class FileHandler {

	public String currentDirectory = System.getProperty("user.dir");
	public File currentDirFile = new File(currentDirectory);
	public String PATH =
		currentDirFile + File.separator + "data" + File.separator + "Missing_data" + File.separator;

	public String extractDatasetRoot(String datasetName) {
		String normalized = datasetName;
		if (normalized.endsWith(".csv")) {
			normalized = normalized.substring(0, normalized.length() - 4);
		}
		String[] parts = normalized.split("_");
		if (parts.length <= 1) {
			return normalized;
		}
		int suffixParts = 4;
		if (parts[parts.length - 1].startsWith("seed")) {
			suffixParts = 5;
		}
		if (parts.length <= suffixParts) {
			return parts[0];
		}
		StringBuilder dataset = new StringBuilder();
		for (int i = 0; i < parts.length - suffixParts; i++) {
			if (i > 0) {
				dataset.append("_");
			}
			dataset.append(parts[i]);
		}
		return dataset.toString();
	}

	public String resolveInputPath(String datasetName) {
		if (datasetName.contains(File.separator) || datasetName.contains("/") || datasetName.contains("\\")) {
			return datasetName;
		}
		String normalized = datasetName;
		if (normalized.endsWith(".csv")) {
			normalized = normalized.substring(0, normalized.length() - 4);
		}
		String dataset = extractDatasetRoot(normalized);
		return PATH + dataset + File.separator + normalized + ".csv";
	}

	public Database readData(String input) {
		Database db = new Database();
		ArrayList<Cell> cells = new ArrayList<>();
		String inputPath = resolveInputPath(input);
		
		try {
			FileReader fr = new FileReader(inputPath);
			BufferedReader br = new BufferedReader(fr);

			String line = null;
			String[] vals = null;

			int rowIndex = 0;
			int num = -1;
			int attrNum = -1;
			boolean[] isNumerical = null;
			double[] minVals = null;
			double[] maxVals = null;
			while ((line = br.readLine()) != null) {
				if (num == -1){
					attrNum = line.split(",").length;
					db.setAttrNum(attrNum);
					num = num + 1;
					isNumerical = new boolean[attrNum];
					minVals = new double[attrNum];
					maxVals = new double[attrNum];
					for (int attri = 0; attri < attrNum; attri++) {
						isNumerical[attri] = true;
						minVals[attri] = Double.MAX_VALUE;
						maxVals[attri] = -Double.MAX_VALUE;
					}
				}
				vals = line.split(",");
				String[] data = new String[attrNum];
				for (int i = 0; i < attrNum; ++i) {
					if (i >= vals.length || vals[i].equals("") || vals[i].equals("nan")) {
						Position pos = new Position(rowIndex, i);
						Cell cell = new Cell(pos);
						cells.add(cell);
					} else {
						data[i] = vals[i].trim();
						if (!Assist.isNumber(data[i])) {
							isNumerical[i] = false;
						} else {
							double tmpVal = Double.parseDouble(data[i]);
							if (tmpVal < minVals[i]) {
								minVals[i] = tmpVal;
							}
							if (tmpVal > maxVals[i]) {
								maxVals[i] = tmpVal;
							}
						}
					}
				}
				Tuple tp = new Tuple(attrNum);
				tp.buildTuple(rowIndex, data);
				db.addTuple(tp);
				rowIndex++;
			}
			db.setIsNumerical(isNumerical);
			db.setMaxVals(maxVals);
			db.setMinVals(minVals);
			int size = db.getLength();
			db.setFlags(new int[size][attrNum]);
			db.setCells(cells);
			br.close();
			fr.close();
		} catch (FileNotFoundException e) {
			throw new RuntimeException("Input file not found: " + inputPath, e);
		} catch (IOException e) {
			throw new RuntimeException("Failed to read input file: " + inputPath, e);
		}
		db.clear();
		return db;
	}

}
