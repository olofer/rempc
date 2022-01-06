%
% Rebuild MPC code
%

isOctave = (exist('OCTAVE_VERSION', 'builtin') ~= 0);
isUnix = isunix();  % copy or cp system commands ?

mex_source = 'qpmpclti2f';
mex_src_ext = 'c';
mex_obj_ext = mexext();

% Clear up old object

mex_obj_name = sprintf('%s.%s', mex_source, mex_obj_ext);
fprintf(1, '\"%s\" : ', mex_obj_name);
if exist(mex_obj_name, 'file') ~= 0
  if isOctave
    autoload(mex_source, file_in_loadpath(mex_obj_name), 'remove');
  end
  delete(mex_obj_name);
  fprintf(1, 'removed\n');
else
  fprintf(1, 'n/a\n');
end

% Recompile

mex_src_file = sprintf('%s.%s', mex_source, mex_src_ext);
fprintf(1, 'compiling source: \"%s\"\n', mex_src_file);

if isOctave
  % [output_, status_] = mkoctfile('--mex', '--verbose', '--strip', '-llapack', '-lopenblas', mex_src_file);
  [output_, status_] = mkoctfile('--mex', '--verbose', '--strip', mex_src_file);
  if status_ ~= 0
    disp(output_);
  end
else
  if isUnix
    status_ = mex(mex_src_file, '-lrt');
  else
    status_ = mex(mex_src_file);
  end
  % status_ = mex('-R2017b', mex_src_file, '-lmwlapack', '-lmwblas');
  %if verLessThan('matlab', '9.3.0')
  %  status_ = mex(mex_src_file);
  %else
  %  status_ = mex('-R2017b', mex_src_file);
  %end
end

main_obj_name = sprintf('%s.%s', mex_source, mex_obj_ext);
isBuildOK = (exist(main_obj_name, 'file') ~= 0) && (status_ == 0);

if ~isBuildOK
  warning('build failed');
  return;
end
