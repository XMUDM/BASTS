package test;

import de.hunsicker.jalopy.Jalopy;


import java.io.File;

public class JavaFormat {
    public static void main(String[] args) {
        for(int i = 1; i <= 30000; i++){
            try {
                Jalopy j = new Jalopy();
                j.setEncoding("utf-8");
                j.setInput(new File("/Users/chao/ASE2020/code_process/process_instruction/code/java_files/" + i +".java"));
                j.setOutput(new File("/Users/chao/ASE2020/code_process/process_instruction/code/java_files_format/" + i +".java"));
                j.format();
            } catch (Exception e) {
//                e.printStackTrace();
            }
        }
    }
}