# data preprocess
Java dataset preprocess

**Note:** You need to change the path in the .py file to your own file path.

## How to split code?
1. code sequence transform to .java file
    * run the method `data2java` of `gen_cfg.py`
    * run `JavaFormat.java` to format the java code

2. generate the CFG of code
    * open .java file in scitools understand(https://www.scitools.com/)
    * run test.pl to generate origin cfg file, save as .txt
    * run the method `process_cfg` of `gen_cfg.py` to get the json format of cfg
    
3. Code splitting according to CFG structure
    * run `dominator_tree.py`
      
        (Convert CFG to dominator tree, split the tree, split the code according to the result of tree splitting, and save it as a txt file, separate different code segments with <sep>)
       
4. Add a method header to the segmentation code segment
    * run the method `add_head` of `gen_cfg.py`
    
5. Generate AST for all code fragments (tool: javalang)
    * run the method `gen_all_ast` of `get_ast.py` to save ast as json format, One json array per line.