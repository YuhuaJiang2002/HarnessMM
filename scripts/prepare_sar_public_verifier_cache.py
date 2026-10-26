"""Verified public fixtures after actor exit, before unchanged native scoring."""
from pathlib import Path
import hashlib,json,shlex
from phiagent.data_engine.provenance import write_json_atomic
BASE=Path('/path/to/experiment-data/figure23')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
async def prepare(environment,directory,task):
 task=Path(task);task_id=task.name
 test=task/'tests/test.sh';test_sha=sha(test)
 # Read only the fixed bootstrap URL after actor completion; no scorer contents reach an actor.
 uses_uv095='https://astral.sh/uv/0.9.5/install.sh' in test.read_text()
 verifier_updates={'UV_DOWNLOAD_URL':'file:///opt/phiagent-uv-cache'} if uses_uv095 else {}
 async def execute(command):
  result=await environment.exec(command=command,timeout_sec=300)
  if result.return_code:raise RuntimeError('Verified public fixture preparation failed')
  return result
 await execute('mkdir -p /opt/phiagent-public-wheels /opt/phiagent-uv-cache /opt/phiagent-official-python3139')
 uv=BASE/'q4-native-uv-095-deps-20261005-v1';uv_manifest=load(uv/'PREPARED.json')
 assert uv_manifest['native_version']=='0.9.5'
 for filename,expected in [('uv-install.sh',uv_manifest['installer_sha256']),('uv-x86_64-unknown-linux-gnu.tar.gz',uv_manifest['archive_sha256'])]:
  assert sha(uv/filename)==expected
  await environment.upload_file(uv/filename,'/opt/phiagent-uv-cache/'+filename)
 envfile=Path(__file__).parent/'official-uv-cache.env'
 await environment.upload_file(envfile,'/opt/phiagent-uv-cache/cache.env')
 python=BASE/'20261005-official-python3139-cache-v1';py_manifest=load(python/'MANIFEST.json');archive=python/py_manifest['asset']['name'];assert sha(archive)==py_manifest['asset']['digest'].removeprefix('sha256:')
 await environment.upload_file(archive,'/opt/phiagent-official-python3139/python.tar.gz')
 await execute('tar -xzf /opt/phiagent-official-python3139/python.tar.gz --strip-components=1 -C /opt/phiagent-official-python3139; ln -sf /opt/phiagent-official-python3139/bin/python3.13 /usr/local/bin/python3.13')
 version=await execute('/usr/local/bin/python3.13 --version');assert version.stdout.strip()=='Python 3.13.9'
 wheels=BASE/'q4-native-scorer-deps-20261005-v1';wheel_manifest=load(wheels/'VERIFIED.json')
 for item in wheel_manifest['files']:
  file=wheels/'wheels'/item['filename'];assert sha(file)==item['sha256'];await environment.upload_file(file,'/opt/phiagent-public-wheels/'+item['filename'])
 apt_manifest=None
 if task_id=='headless-terminal':
  verifier_updates['APT_CONFIG']='/opt/phiagent-native-dependency-cache/apt.conf'
  public=BASE/'q4-paid-verifier-20261005-v37-newparent-headless-native/source';apt_manifest=load(public/'OFFLINE_NATIVE_DEPENDENCIES.json');assert apt_manifest['native_apt_signed_index_verified']is True
  await execute('mkdir -p /opt/phiagent-native-dependency-cache/debs /opt/phiagent-native-dependency-cache/empty-sources /opt/phiagent-native-dependency-cache/empty-lists; touch /opt/phiagent-native-dependency-cache/empty.sources')
  await environment.upload_file(public/'offline-apt.conf','/opt/phiagent-native-dependency-cache/apt.conf')
  for package,item in apt_manifest['packages'].items():
   assert sha(item['transport_path'])==item['SHA256'];await environment.upload_file(Path(item['transport_path']),'/opt/phiagent-native-dependency-cache/debs/'+package+'.deb')
  await execute('dpkg -i /opt/phiagent-native-dependency-cache/debs/*.deb')
  for package,item in apt_manifest['packages'].items():
   checked=await execute("dpkg-query -W -f='${Status}|${Version}' "+shlex.quote(package));assert checked.stdout=='install ok installed|'+item['Version']
 assert sha(test)==test_sha
 write_json_atomic(Path(directory)/'VERIFIED_PUBLIC_VERIFIER_FIXTURES.json',{'actor_calls_added':0,'actor_already_finished':True,'native_timer_not_started':True,'task_and_test_bytes_changed':False,'official_uv':uv_manifest,'official_python':py_manifest,'official_wheels':wheel_manifest,'official_apt':apt_manifest,'cache_env_sha256':sha(envfile),'native_test_sha256':test_sha,'verifier_environment_updates':verifier_updates})
 return verifier_updates
