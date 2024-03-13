data_path = '/PATH/TO/MAIN-FOLDER/';

fname = '/PATH/TO/MAIN-FOLDER/to_path.json';
fid = fopen(fname);
raw = fread(fid, inf);
str = char(raw);
fclose(fid);
val = jsondecode(str);

% Iterate over track while parsing and storing the hierarchical matrix
fn = fieldnames(val);
for k=1:numel(fn)
    filename = val.(fn{k});
    filename = append(data_path, filename(3:end), '.sib_sing.mid');
    [nmat, mstr] = readmidi(filename);
    mh = metrichierarchy(nmat);
    output_filename = append(filename(1:end-3), 'mat')
    save(output_filename, 'mh')
end

