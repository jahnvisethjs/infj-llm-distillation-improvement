"""Create an evaluation-only recovery copy; the original helper is not edited."""
import ast


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Source differs from reviewed evaluation runner: '+old[:65])
    return source.replace(old,new,1)


def patch_runner(source):
    source=replace_once(source,"from select_b import (ORTOOLS_VERSION, VERSION, bounds, cell, csv_bytes, digest,",
                        "from evaluation_recovery import archive_prediction_bundle, enter\n\nfrom select_b import (ORTOOLS_VERSION, VERSION, bounds, cell, csv_bytes, digest,")
    old="""                if pred.exists() or marker.exists():
                    raise FileExistsError('Partial/different predictions were preserved: '+str(pred)+
                                          '. Inspect them before starting a separately named retry.')
"""
    new="""                if pred.exists() or marker.exists() or pred.with_suffix('.config.json').exists():
                    archive_prediction_bundle(root,pred)
"""
    source=replace_once(source,old,new)
    old="""            if valid:
                validate_pair(pred,pred)
"""
    new="""            if valid:
                try:
                    saved_runtime=read(pred.with_suffix('.config.json'))
                    valid=(saved_runtime.get('gpu')==torch.cuda.get_device_name(0) and
                           saved_runtime.get('torch')==torch.__version__ and
                           saved_runtime.get('cuda')==torch.version.cuda)
                except (OSError,ValueError):
                    valid=False
            if valid:
                validate_pair(pred,pred)
"""
    source=replace_once(source,old,new)
    old="""            if marker.is_file() and pred.is_file():
                prior=read(marker)
                valid=prior.get('signature')==sig and prior.get('predictions_sha256')==digest(pred) and prior.get('config_sha256')==digest(pred.with_suffix('.config.json'))
"""
    new="""            if marker.is_file() and pred.is_file():
                try:
                    prior=read(marker)
                    valid=prior.get('signature')==sig and prior.get('predictions_sha256')==digest(pred) and prior.get('config_sha256')==digest(pred.with_suffix('.config.json'))
                except (OSError,ValueError,KeyError):
                    valid=False
"""
    source=replace_once(source,old,new)
    if source.count("HERE/'run_method_b.py'")!=2:
        raise ValueError('Unexpected number of subprocess script references')
    source=source.replace("HERE/'run_method_b.py'","Path(__file__).resolve()")
    source=replace_once(source,"if __name__=='__main__':main()","if __name__=='__main__':enter(main)")
    ast.parse(source)
    return source
