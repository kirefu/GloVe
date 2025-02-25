
# nohup ~/.local/bin/snakemake --snakefile ../../Snakefile --cluster "sbatch --nice {cluster.oversubscribe} {cluster.gres}" --cluster-config ../cluster.json --cores 4222 -j 4222 -k --configfile  config.pt-en.yaml  &

import sys
import glob
import gzip
import lzma
import tldextract
import os
import os.path
import socket
import shutil
from tld import get_tld
from tqdm import tqdm
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from toolwrapper import ToolWrapper
from func_timeout import func_timeout, FunctionTimedOut
from cerberus import Validator

###########################################################
# UTILS

def binaryAvailable(cmd):
    cmd="command -v "+cmd+" > /dev/null"
    callout=os.system(cmd)
    if callout == 0:
        return True
    else:
        return False

def tokeniserCheck(cmd):
    proc = ToolWrapper(cmd.split())
    line = proc.writeline('test.test')
    try:
        tokline = func_timeout(5, proc.readline)
    except FunctionTimedOut:
        sys.stderr.write("ERROR: tokeniser could not complete within 5 seconds and was terminated. Is it buffering stdout? (if you are using Moses tokeniser, add -b)\n")
        exit(1)

def systemCheck(cmd):
    #sys.stderr.write("Executing:" + cmd + " on " + socket.gethostname() + "\n")
    #sys.stderr.flush()

    subprocess.check_call(cmd, shell=True)

@contextmanager
def open_gzip_or_plain(file_path):

    def decode_text(file_handler):
        for line in file_handler:
            yield line.decode('utf-8')

    f = None
    try:
        #print("file_path", file_path)
        if file_path[-3:] == ".gz":
            f = gzip.open(file_path, 'rb')
            yield decode_text(f)
        elif file_path[-3:] == ".xz":
            f = lzma.open(file_path, 'rb')
            yield decode_text(f)
        else:
            f = open(file_path, 'r')
            yield f

    except Exception as ex:
        sys.stderr.write(str(ex)+"\n")
        raise Exception("Error occured while loading a file {}".format(file_path))

    finally:
        if f:
            f.close()


def ValidateArgs(config):
    schema = {'bitextor': {'required': True, 'type': 'string'},
            'lang1': {'required': True, 'type': 'string', 'maxlength': 2},
            'lang2': {'required': True, 'type': 'string', 'maxlength': 2},
            'temp': {'type': 'string'},
            'hunalignThreshold': {'type': 'float'},
            'maxlines': {'type': 'integer'},
            'alcazar': {'type': 'boolean'},
            'onlyConcat': {'type': 'boolean'},
            'profiling': {'type': 'boolean'},

           'permanentDir': {'required': True, 'type': 'string'},
           'transientDir': {'required': True, 'type': 'string'},
           'boilerpipeCleaning': {'type': 'boolean'},
           'pdf-converter' : {'type' : 'string'},
           'httrack': {'type': 'boolean'}, #deprecated
           'crawler': {'type': 'string'},
           'crawlTld': {'type': 'boolean'},
           'crawlerNumThreads': {'type': 'integer'},
           'maxSizeWARC': {'type': 'integer'},

           'dic': {'type': 'string'},
           'LANG1Tokenizer': {'type': 'string'},
           'LANG2Tokenizer': {'type': 'string'},
           'LANG2Detokenizer': {'type': 'string'},

           'LANG1SentenceSplitter': {'type': 'string'},
           'LANG2SentenceSplitter': {'type': 'string'},

           'crawlerUserAgent': {'type': 'string'},
           'crawlSizeLimit': {'type': 'string'},
           'crawlTimeLimit': {'type': 'string'},
           'crawlWait': {'type': 'integer'},
           'crawlFileTypes': {'type': 'string'},
           'crawlPageLimit': {'type': 'integer'},
           'crawlerConnectionTimeout': {'type': 'integer'},
           'dumpCurrentCrawl': {'type': 'string'},
           'resumePreviousCrawl': {'type': 'string'},

           'documentAligner': {'type': 'string'},
           'mosesDir': {'type': 'string'},
           'alignerCmd': {'type': 'string'},
           'bleualign': {'type': 'boolean'},
           'docAlignThreshold': {'type': 'float'},
           'bleuAlignThreshold': {'type': 'float'},

           'bicleaner': {'type': 'string'},
           'bicleanerThreshold': {'type': 'float'},
           'elrc': {'type': 'boolean'},
           'restorative': {'type': 'boolean'},
           'deferredCrawling': {'type': 'boolean'},

           'deduped': {'type': 'boolean'},
           'hosts': {'type': 'list'},
           'hostPath': {'type': 'string'},
           'excludeHosts': {'type': 'list'},
           'excludeHostsFile': {'type': 'string'},
           'linkedHosts': {'type': 'list'},
           'linkedHostsAction': {'type': 'string'},

           'langstat': {'type': 'string'},
           'langstatExcludeStrings': {'type': 'string'},
           'langstatThreshold': {'type': 'integer'},

           'initCorpusTrainPrefix': {'type': 'list'},
           'initCorpusDevPrefix': {'type': 'list'},
           'initCorpusTestPrefix': {'type': 'list'},

           'bicleanerCorpusTrainingPrefix': {'type': 'list'},

           'nmt': {'type': 'boolean'},
           'smt': {'type': 'boolean'},
           'storeRawCorpus': {'type': 'boolean'},
           'tmx': {'type': 'boolean'},

           'gpuId': {'type': 'integer'},
           'marianDir': {'type': 'string'},
           'marianArgs': {'type': 'list'},
           'marianModelFile': {'type': 'string'},
           'nmtVocabSize': {'type': 'integer'},

           'subwordNmtDir': {'type': 'string'},

           'mgiza': {'type': 'string'},

            }

    if ("httrack" in config and config["httrack"] == True) or ("crawler" in config and config["crawler"] == 'httrack'):
        if not binaryAvailable("httrack"):
            sys.stderr.write("WARNING: HTTrack is not installed. Install it or disable option 'httrack' in the configuration file.\n")

    config.update({k: os.path.expanduser(v) if isinstance(v, str) else v for k, v in config.items()})
    config.update({k: [ os.path.expanduser(el) for el in v ] if 'Config' in k and v is list else v for k, v in config.items()})

    #Mandatory options depending on the document aligner method choosen
    if "documentAligner" in config:
        if config["documentAligner"]=='NMT':
            schema['marianDir']['required']=True
            schema['subwordNmtDir']['required']=True
            schema['mosesDir']['required']=True
            schema['LANG2Detokenizer']['required']=True
            schema['nmtVocabSize']['required']=True
            schema['gpuId']['required']=True
            schema['marianArgs']['required']=True
            schema['initCorpusTrainPrefix']['required']=True
            schema['initCorpusDevPrefix']['required']=True
            schema['initCorpusTestPrefix']['required']=True
        elif config["documentAligner"]=='SMT':
            schema['mosesDir']['required']=True
        elif config["documentAligner"]=='externalMT':
            schema['alignerCmd']['required']=True
        else:
            schema['dic']['required']=True
            if "dic" in config and not os.path.isfile(config["dic"]):
                schema['initCorpusTrainPrefix']['required']=True
    else:
        schema['dic']['required']=True
        if "dic" in config and not os.path.isfile(config["dic"]):
            schema['initCorpusTrainPrefix']['required']=True

    if "bicleaner" in config:
        if not os.path.isfile(config["bicleaner"]):
            schema['bicleanerCorpusTrainingPrefix']['required']=True

    if "linkedHosts" in config:
        schema['linkedHostsAction']['required']=True

    v = Validator(schema)
    #v.allow_unknown = True

    b = v.validate(config)

    if not b:
        print("Validation error. Stopping.", v.errors)
        exit()

###########################################################
sys.stderr.write("Starting bitextor\n")

ValidateArgs(config)

#Local bitextor installation
BITEXTOR=config["bitextor"]

#Crawled languages
LANG1=config["lang1"]
LANG2=config["lang2"]

#Working paths
permanent=config["permanentDir"]
transient=config["transientDir"]

PROFILING=""

if "temp" in config:
  TMPDIR=config["temp"]
else:
  TMPDIR=transient
shell("mkdir -p "+TMPDIR)

if "hunalignThreshold" in config:
  MINQUALITY=config["hunalignThreshold"]
else:
  MINQUALITY=0.0

if "maxlines" in config:
  MAXLINES=config["maxlines"]
else:
  MAXLINES=-1

if "alcazar" in config and config["alcazar"]:
  ALCAZAR = "--alcazar"
else:
  ALCAZAR = ""

if "profiling" in config and config["profiling"]:
  PROFILING="/usr/bin/time -v"

systemCheck("mkdir -p " + permanent)
systemCheck("mkdir -p " + transient)

#Dictionary
if "dic" in config:
  DIC=config["dic"]
else:
  DIC=None

#Option to remove Boilerpipe html: if the option is enabled, boilerpipe is not used
if "boilerpipeCleaning" in config and config["boilerpipeCleaning"]==True:
  boilerpipeCleaning = '--boilerpipe'
else:
  boilerpipeCleaning = ''

if "pdf-converter" in config and config["pdf-converter"]=="pdf-extract":
  usepdfextract = "--pdfextract"
else:
  usepdfextract = ""

#Option to use HTTrack for crawling instead of the native bitextor crawler
if "httrack" in config and config["httrack"]==True:
  CRAWLTARGET="httrack"
else:
  CRAWLTARGET="creepy"

if "crawler" in config:
  CRAWLTARGET=config["crawler"]

#Maximum size of the WARC file to be pre-processed: if it is exceeded the file is split
if "maxSizeWARC" in config:
  MAXWARCSIZE="-m "+str(config["maxSizeWARC"])
else:
  MAXWARCSIZE=""

#Tokenisers
if "LANG1Tokenizer" in config:
  WORDTOK1=config["LANG1Tokenizer"]

if "LANG2Tokenizer" in config:
  WORDTOK2=config["LANG2Tokenizer"]

if "LANG1SentenceSplitter" in config:
  SENTTOK1=config["LANG1SentenceSplitter"]

if "LANG2SentenceSplitter" in config:
  SENTTOK2=config["LANG2SentenceSplitter"]

############ OPTIONS FOR THE NATIVE BITEXTOR CRAWLER ############

#If this option is enabled the crawler will keep crawling across a whole top-level domain (.es, .com, .fr, etc.)
if "crawl-tld" in config and config["crawl-tld"]:
  TLD_CRAWL="-D"
else:
  TLD_CRAWL=""

#If this option is set, a specific user agent is used when crawling
if "crawlerUserAgent" in config:
  USERAGENT="-a \""+config["crawlerUserAgent"]+"\""
else:
  USERAGENT=""

#If this option is enabled, a size-limit is set for crawled data (for example "size-limit": "1G")
if "crawlSizeLimit" in config:
  CRAWLSIZELIMIT="-s "+config["crawlSizeLimit"]
else:
  CRAWLSIZELIMIT=""

#If this option is enabled, a time-limit is set for crawling data (for example "time-limit": "1h")
if "crawlTimeLimit" in config:
  CRAWLTIMELIMIT="-t "+str(config["crawlTimeLimit"])
else:
  CRAWLTIMELIMIT=""

if "crawlWait" in config:
  CRAWLWAIT="--wait "+str(config["crawlWait"])
else:
  CRAWLWAIT=""
if "crawlPageLimit" in config:
  CRAWLPAGELIMIT="-p "+str(config["crawlPageLimit"])
else:
  CRAWLPAGELIMIT=""

if "crawlFileTypes" in config:
  CRAWLFILETYPES="-f "+str(config["crawlFileTypes"])
else:
  CRAWLFILETYPES=""

#Option to set how many threads will be used for crawling (default value: 2). Note that too many threads can cause the server hosting the website to reject some of the simultaneous connections.
if "crawlerNumThreads" in config:
  CRAWLJOBS="-j "+str(config["crawlerNumThreads"])
else:
  CRAWLJOBS="-j 2"

#Connection timeout in the crawler
if "crawlerConnectionTimeout" in config:
  CRAWLTIMEOUT="-o "+str(config["crawlerConnectionTimeout"])
else:
  CRAWLTIMEOUT=""

#If this option is set, the "crawler" object will be dump as a pickle, so crawling can be continued afterwards
if "dumpCurrentCrawl" in config:
  CRAWLDUMPARGS="-d "+config["dumpCurrentCrawl"]
else:
  CRAWLDUMPARGS=""

#If this option is set, crawling will be continued from the pickle object dumped in a previous crawl
if "resumePreviousCrawl" in config:
  CONTINUECRAWL="-l "+config["resumePreviousCrawl"]
else:
  CONTINUECRAWL=""


############ OPTIONS FOR THE MT DOCUMENT ALIGNER ############

#If documentAligner is enabled, Marek Střelec's MT-based document aligner is used (bitextor/document-aligner)
if "documentAligner" in config:
  if config["documentAligner"] == "externalMT":
    MT_COMMAND=config["alignerCmd"]
    DOCALIGNEXT="customMT"
  elif config["documentAligner"] == "SMT":
    MT_COMMAND="smt"
    DOCALIGNEXT="smt"
  elif config["documentAligner"] == "NMT":
    MT_COMMAND="nmt"
    DOCALIGNEXT="nmt"
  else:
    MT_COMMAND=""
    DOCALIGNEXT="bitextor"
else:
  MT_COMMAND=""
  DOCALIGNEXT="bitextor"

if "mosesDir" in config:
  MOSESDIR = config["mosesDir"]
else:
  MOSESDIR = ""

if "bleualign" in config and config["bleualign"]:
  SEGMENTALIGNER="bleualign"
else:
  SEGMENTALIGNER="hunalign"

# Marek says DOC_THRESHOLD~0.1, BLEU_THRESHOLD~ 0.1 - 0.3
if "docAlignThreshold" in config:
    DOC_THRESHOLD  = config["docAlignThreshold"]
else:
    DOC_THRESHOLD = 0.1

if "bleuAlignThreshold" in config:
    BLEU_THRESHOLD = config["bleuAlignThreshold"]
else:
    BLEU_THRESHOLD = 0.2
#print("DOC_THRESHOLD", DOC_THRESHOLD, "BLEU_THRESHOLD", BLEU_THRESHOLD)

############ FILTERING AND POST-PROCESSING OPTIONS ############

if "bicleaner" in config:
  RAWOPTION="bicleaner.scores"
  BICLEANEROPTION=",bicleaner"
  BICLEANER="bicleaner"
  BICLEANER_CONFIG=config["bicleaner"]
  BICLEANER_SORT="-r -k6,6 -k3,4"
  tokeniserCheck(WORDTOK1)
  tokeniserCheck(WORDTOK2)

else:
  RAWOPTION="segclean"
  BICLEANEROPTION=""
  BICLEANER_SORT="-k3,4"
  BICLEANER="segclean"
  BICLEANER_CONFIG=""

if "bicleanerThreshold" in config:
  BICLEANER_THRESHOLD=config["bicleanerThreshold"]
else:
  BICLEANER_THRESHOLD=0.0

if "elrc" in config and config["elrc"]:
  ELRCSCORES="elrc"
  ELRCFIELDS=",lengthratio,numTokensSL,numTokensTL"
else:
  ELRCSCORES=BICLEANER
  ELRCFIELDS=""

if "deferredCrawling" in config and config["deferredCrawling"]:
  DEFERREDFIELDS=",deferredseg1,checksum1,deferredseg2,checksum2"
  DEFERREDSENTENCES=".deferred"
else:
  DEFERREDFIELDS=""
  DEFERREDSENTENCES=""

if "restorative" in config and config["restorative"]:
  RESTORATIVE="restorative"
else:
  RESTORATIVE="segclean"

#========================= MAPPING URLS AND OUTPUT FILES =========================#

def CreateDomainKey2HostMap(hosts):
    ret={}
    for host in hosts:
        # don't merge blog sites
        if host.find(".blogspot.") >= 0 or host.find(".wordpress.") >= 0:
           key = host
        else:
           key = tldextract.extract(host).domain

        if key not in ret:
            ret[key]=[]
        ret[key].append(host)
        #print("subdomain", key, host)
    return ret

def FilterTLD(tlds):
    filtered_tlds={}
    if os.path.isfile("{permanent}/domains.gz".format(permanent=permanent)):
        with open_gzip_or_plain("{permanent}/domains.gz".format(permanent=permanent)) as f:
            for tld in f:
                tld=tld.strip()
                filtered_tlds[tld]=tlds[tld]
        return filtered_tlds
    else:
        return tlds

def InitConcatLogicLink(domains):
    for tld,hosts in domains.items():
        if len(hosts) == 1 and  os.path.isfile("{permanent}/warc/{host}/{crawler}.warc.xz".format(permanent=permanent, host=hosts[0], crawler=CRAWLTARGET)) and not os.path.isfile("{transient}/{tld}/concat.warc.xz".format(tld=tld,transient=transient)):
            cmd="mkdir -p {transient}/{tld}; ln -sfn {permanent}/warc/{host}/{crawler}.warc.xz {transient}/{tld}/concat.warc.xz".format(permanent=permanent, host=hosts[0], crawler=CRAWLTARGET, transient=transient,
 tld=tld)
            shell(cmd)

def LoadDomains(file_path):
    domains = set()
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            if len(line):
                domains.add(line)

    return domains

def GetDomainKeys(hosts):
    keys = set()
    for host in hosts:
        domain = tldextract.extract(host).domain
        keys.add(domain)
    return keys

def ExcludeHosts(hosts, excludeHosts):
    excludeKeys = GetDomainKeys(excludeHosts)

    hostsCopy = set(hosts)
    print("BEFORE hosts", len(hosts), len(hostsCopy))

    for host in hostsCopy:
        key = tldextract.extract(host).domain
        if key in excludeKeys:
            hosts.remove(host)
    print("AFTER hosts", len(hosts), len(hostsCopy))

def GetHostsFromLangstat(langstat_path, lang1, lang2, threshold, exclude_path):
    print("langstat_path", langstat_path, file=sys.stderr)
    l12 = [lang1.lower(), lang2.lower()]

    excluded_set = set()
    if exclude_path:
        excluded_set = LoadDomains(Path(exclude_path))

    hostsToCrawl = set()

    #sys.stderr.write(
    #    "Gathering domain information for {0} and {1}...\n".format(*l12))
    with tqdm(total=None) as pbar:
        with open_gzip_or_plain(langstat_path) as f:

            prevHost = ""
            langContent = {}

            for line in f:
                split_line = line.strip().split()
                if len(split_line) != 3:
                    continue

                host, lang, byte_len = split_line
                name = tldextract.extract(host).domain
                #print("processing ", host, lang.lower(), byte_len, name)

                if host != prevHost:
                    # start of new host. Process previous entries
                    if len(langContent) == 2:
                        lang1_bytes = langContent[l12[0]]
                        lang2_bytes = langContent[l12[1]]
                        if lang1_bytes >= threshold and lang2_bytes >= threshold:
                            hostsToCrawl.add(prevHost)

                    prevHost = host
                    langContent = {}

                if lang.lower() in l12 and name not in excluded_set:
                    langContent[lang.lower()] = int(byte_len)

                pbar.update(1)

            # last host
            if len(langContent) == 2:
                lang1_bytes = langContent[l12[0]]
                lang2_bytes = langContent[l12[1]]
                if lang1_bytes >= threshold and lang2_bytes >= threshold:
                    hostsToCrawl.add(prevHost)

    return hostsToCrawl

def LinkedHosts(permanent, dir, hosts, linkedHostsAction):
    #print("dir", dir)
    cmd = "mkdir -p " + permanent + "/warc"
    shell(cmd)

    with gzip.open(dir + "/hosts.gz", 'rt') as f:
        otherHosts = f.read().splitlines()

    otherHosts = set(otherHosts)
    #print("otherHosts", otherHosts)

    # make a copy in case we have to delete thing
    copyHosts = set(hosts)
    for host in copyHosts:
        if host in otherHosts:
            if linkedHostsAction == "remove" or linkedHostsAction == "postCrawlExclude":
                hosts.remove(host)
            elif linkedHostsAction == "link":
                dest = "{permanent}/warc/{host}".format(permanent=permanent, host=host)

                if not (os.path.exists(dest) or os.path.islink(dest)):
                    cmd = "ln -sfn {dir}/warc/{host} {dest}".format(dir=dir, host=host, dest=dest)
                    #print(cmd)
                    shell(cmd)
            else:
                sys.stderr.write("Unknown linkedHostsAction:" + linkedHostsAction + "\n")
                exit()

def PostCrawlExclude(hosts, postCrawlPath):
    with open(postCrawlPath, "rt") as f:
        excludeList = f.read().splitlines()
    for exclude in excludeList:
        exclude = exclude.strip()
        if len(exclude) > 0 and exclude in hosts:
            hosts.remove(exclude)

###############################################################################################
if os.path.isfile(permanent + "/hosts.gz"):
    with gzip.open(permanent + "/hosts.gz", 'rt') as f:
        hosts = f.read().splitlines()
    hosts = set(hosts)
    #sys.stderr.write("read hosts from file=" + str(len(hosts)) + "\n")

else:
    hosts = set()

    if "hosts" in config:
        newHosts = config["hosts"]
        hosts = hosts.union(newHosts)
        #sys.stderr.write("#hosts given=" + str(len(newHosts)) + "\n")

    if "langstat" in config:
        langstat_path = config["langstat"]
        lang1 = config["lang1"]
        lang2 = config["lang2"]
        threshold = int(config["langstatThreshold"])
        exclude_path = config["langstatExcludeStrings"]

        newHosts = GetHostsFromLangstat(langstat_path, lang1, lang2, threshold, exclude_path)
        hosts = hosts.union(newHosts)
        #sys.stderr.write("#hosts found in langstat=" + str(len(newHosts)) + "\n")

    if "hostPath" in config:
        path = config["hostPath"]
        with gzip.open(path, 'rt') as f:
            newHosts = f.read().splitlines()
            #sys.stderr.write("#hostPath=" + str(len(newHosts)) + "\n")

        hosts = hosts.union(newHosts)

    if "excludeHosts" in config:
        excludeHosts = config["excludeHosts"]
        ExcludeHosts(hosts, excludeHosts)

    if "excludeHostsFile" in config:
        with open(config["excludeHostsFile"], "rt") as f:
            excludeHosts = f.read().splitlines()
            print("excludeHosts", len(excludeHosts))
        ExcludeHosts(hosts, excludeHosts)

    if (len(hosts) == 0):
        print("No hosts found. Need at least one of hosts, langstat, hostPath")
        exit()

    with gzip.open(permanent + "/hosts.gz", 'wt') as f:
        for host in hosts:
            f.write("%s\n" % host)

sys.stderr.write("#hosts=" + str(len(hosts)) + "\n")

# exclude dead domains
postCrawlPath = "{permanent}/post-crawl-exclude".format(permanent=permanent)
#print("postCrawlPath", postCrawlPath)
if os.path.isfile(postCrawlPath):
    PostCrawlExclude(hosts, postCrawlPath)

if "linkedHosts" in config:
    linkedHostsAction = config["linkedHostsAction"]

    if linkedHostsAction != "postCrawlExclude":
        # exclude dead domains in other language pairs
        for dir in config["linkedHosts"]:
            postCrawlPath = "{dir}/permanent/post-crawl-exclude".format(dir=dir)
            #print("postCrawlPath", postCrawlPath)
            if os.path.isfile(postCrawlPath):
                PostCrawlExclude(hosts, postCrawlPath)

    for dir in config["linkedHosts"]:
        LinkedHosts(permanent, dir, hosts, linkedHostsAction)

    if linkedHostsAction == "postCrawlExclude":
        # create list of dead domains in post-crawl-exclude file, only for this language pair
        print("Calc post-crawl exclude")
        #hostsPath = permanent + "/hosts.gz"
        #with gzip.open(hostsPath, 'rt') as f:
        #    hosts = f.read().splitlines()
        #hosts = set(hosts)
	#print("hosts", hosts)

        warcPath = permanent + "/warc"
        domainDirs = os.listdir(warcPath)
        for domainDir in domainDirs:
            #print(domainDir)
            warcFilePath = "{warcPath}/{domainDir}/httrack.warc.xz".format(warcPath=warcPath, domainDir=domainDir)
            if os.path.isfile(warcFilePath):
               hosts.remove(domainDir)
        #print("hosts", hosts)

        postCrawlPath = permanent + "/post-crawl-exclude"
        with open(postCrawlPath, 'wt') as f:
            f.write("\n".join(hosts))
        exit()

sys.stderr.write("#hosts to crawl/process=" + str(len(hosts)) + "\n")
#exit(-1)

domainKey2Hosts = CreateDomainKey2HostMap(hosts)
#If file domains.gz exists in the permanent directory, the dictionary domainKey2Hosts is filtered to contain only those TLD in this file
domainKey2Hosts = FilterTLD(domainKey2Hosts)
#Function that checks if a domain has only one WARC and, if so, it creates a symbolic link
InitConcatLogicLink(domainKey2Hosts)
#print("domainKey2Hosts", domainKey2Hosts)

# every shell command will run sync
#shell.prefix("sync; set -euo pipefail; ")
shell.prefix("set -euo pipefail; ")

#================================== Creater Moses EMS config========================#
def CreateMosesEMSConfig(workDir, mosesDir, mgiza, trainPrefixes, devPrefix, testPrefixes):
    with open("{mosesDir}/scripts/ems/example/config.basic.moses2".format(mosesDir=mosesDir), "r") as file:
        lines = file.read().split("\n")
        print("lines", len(lines))
        #print("lines", lines)

    lines[8] = "working-dir = {0}".format(workDir)
    lines[11] = "input-extension = " + LANG1
    lines[12] = "output-extension = " + LANG2
    lines[13] = "#" + lines[13]

    lines[18] = "moses-src-dir = {mosesDir}".format(mosesDir=mosesDir)
    lines[27] = "external-bin-dir = {mgiza}/mgizapp/inst".format(mgiza=mgiza)
    lines[79] = "jobs = 10"
    lines[94] = "cores = 8"
    lines[146] = "settings = \"--prune '0 0 1' -T $working-dir/lm -S 20% --discount_fallback\" "
    lines[310] = "training-options = \"-mgiza -mgiza-cpus 8\""
    lines[384] = "binarize-all = $moses-script-dir/training/binarize-model.perl"

    # DELETES
    lines[30] = ""
    lines[33] = ""
    lines[36] = ""
    lines[39] = ""
    lines[39] = ""

    # corpus
    lines[132] = lines[132] + " IGNORE"
    lines[133] = lines[133] + " IGNORE"

    # lm
    lines[206] = lines[206] + " IGNORE"
    lines[207] = lines[207] + " IGNORE"

    # tuning
    lines[522] = ""
    lines[528] = ""

    # eval
    lines[640] = "#" + lines[640]
    lines[641] = "#" + lines[641]
    lines[645] = 'sacre-bleu = "sacrebleu -lc"'
    lines[646] = 'sacre-bleu-c = "sacrebleu"'
    lines[664] = "#" + lines[664]
    lines[670] = "#" + lines[670]

    lines[682] = lines[682] + " IGNORE"

    # ADD
    # eval
    lines.insert(710, "")

    line = 711
    for path in testPrefixes:
        name = os.path.basename(path)
        lines.insert(line, "[EVALUATION:{name}]".format(name=name))
        lines.insert(line + 1, "raw-input = {path}.{lang}".format(path=path, lang=LANG1))
        lines.insert(line + 2, "raw-reference = {path}.{lang}".format(path=path, lang=LANG2))

        line += 3

    # tuning
    assert(len(devPrefix) == 1)
    lines[523] = "raw-input = {path}.{lang}".format(path=devPrefix[0], lang=LANG1)
    lines[529] = "raw-reference = {path}.{lang}".format(path=devPrefix[0], lang=LANG2)

    # lm
    line = 215
    for path in trainPrefixes:
        name = os.path.basename(path)
        lines.insert(line, "[LM:{name}]".format(name=name))
        lines.insert(line + 1, "raw-corpus = {path}.{lang}".format(path=path, lang=LANG2))

        line += 2

    # corpus
    lines.insert(137, "")

    line = 138
    for path in trainPrefixes:
        name = os.path.basename(path)
        lines.insert(line, "[CORPUS:{name}]".format(name=name))
        lines.insert(line + 1, "raw-stem = {path}".format(path=path))

        line += 2


    with open("{0}/steps/1/config.1".format(workDir), "w") as file:
        file.write("\n".join(lines))



#================================== START SNAKEMAKE================================#

#================================== TARGET FILES ==================================#

OUTPUT=[]

if "onlyConcat" in config and config["onlyConcat"]:
    for tld in domainKey2Hosts.keys():
        OUTPUT.append("{dir}/{tld}/concat.warc.xz".format(dir=transient, tld=tld))
else:
    OUTPUT.append("{dir}/{l1}-{l2}.sent.xz".format(dir=permanent, l1=LANG1, l2=LANG2))
    if "nmt" in config and config["nmt"]:
        OUTPUT.append("{dir}/nmt-dir/evaluation/report".format(dir=transient))
        OUTPUT.append("{dir}/nmt-dir-crawl/evaluation/report".format(dir=transient))

    if "smt" in config and config["smt"]:
        OUTPUT.append("{dir}/smt-dir/steps/1/REPORTING_report.1.DONE".format(dir=transient))
        OUTPUT.append("{dir}/smt-dir-crawl/steps/1/REPORTING_report.1.DONE".format(dir=transient))

    #Optional TMX: if option enabled, TMX is generated; otherwhise, tab-separated .sent file is generated
    if "tmx" in config and config["tmx"]:
        if "storeRawCorpus" in config and config["storeRawCorpus"]:
            OUTPUT.append("{dir}/{l1}-{l2}.raw.xz".format(dir=permanent, l1=LANG1, l2=LANG2))
        if "deduped" in config and config["deduped"]:
            OUTPUT.append("{dir}/{l1}-{l2}.deduped.tmx.xz".format(dir=permanent, l1=LANG1, l2=LANG2))
        else:
            OUTPUT.append("{dir}/{l1}-{l2}.not-deduped.tmx.xz".format(dir=permanent, l1=LANG1, l2=LANG2))
    else:
        OUTPUT.append("{dir}/{l1}-{l2}.sent.xz".format(dir=permanent, l1=LANG1, l2=LANG2))

rule all:
    input:
        expand("{target}", target=OUTPUT)

#================================== SMT ======================================#

rule train_smt_with_crawl_data:
    input:
        l1="{dir}/crawl.{lang}".format(dir=permanent, lang=LANG1)
        ,
        l2="{dir}/crawl.{lang}".format(dir=permanent, lang=LANG2)
        ,
        emsConfig = "{dir}/smt-dir-crawl/steps/1/config.1".format(dir=transient)

    output:
        report = "{dir}/smt-dir-crawl/steps/1/REPORTING_report.1.DONE".format(dir=transient)
    run:
        cmd = "cd {transient}/smt-dir-crawl &&  {MOSESDIR}/scripts/ems/experiment.perl --continue 1 --exec;"
        shell(cmd)

rule train_smt_with_crawl_data_create_config:
    input:
        l1="{dir}/crawl.{lang}".format(dir=permanent, lang=LANG1)
        ,
        l2="{dir}/crawl.{lang}".format(dir=permanent, lang=LANG2)
        ,
        config = "{dir}/config.json".format(dir=transient)

    output:
        emsConfig = "{dir}/smt-dir-crawl/steps/1/config.1".format(dir=transient)
    priority: 40

    run:
        cmd = "mkdir -p {dir}/smt-dir-crawl/steps/1".format(dir=transient)
        shell(cmd)

        trainData = config["initCorpusTrainPrefix"]

        crawlPref = "{dir}/crawl".format(dir=permanent)
        trainData.append(crawlPref)

        CreateMosesEMSConfig("{0}/smt-dir-crawl".format(transient),
                            MOSESDIR, config["mgiza"],
                            trainData,
                            config["initCorpusDevPrefix"],
                            config["initCorpusTestPrefix"])

rule train_smt_all:
    input:
        emsConfig = "{dir}/smt-dir/steps/1/config.1".format(dir=transient)

    output:
        report = "{dir}/smt-dir/steps/1/REPORTING_report.1.DONE".format(dir=transient)
    run:
        cmd = "cd {transient}/smt-dir && {MOSESDIR}/scripts/ems/experiment.perl --continue 1 --exec;"
        shell(cmd)

rule train_smt_all_create_config:
    input:
        config = "{dir}/config.json".format(dir=transient)

    output:
        emsConfig = "{dir}/smt-dir/steps/1/config.1".format(dir=transient)
    priority: 40

    run:
        cmd = "mkdir -p {dir}/smt-dir/steps/1".format(dir=transient)
        shell(cmd)

        CreateMosesEMSConfig("{0}/smt-dir".format(transient),
                            MOSESDIR, config["mgiza"],
                            config["initCorpusTrainPrefix"],
                            config["initCorpusDevPrefix"],
                            config["initCorpusTestPrefix"])

#================================== NMT ======================================#


rule train_nmt_with_crawl_data:
    input:
        l1="{dir}/crawl.{lang}".format(dir=permanent, lang=LANG1)
        ,
        l2="{dir}/crawl.{lang}".format(dir=permanent, lang=LANG2)
        ,
        config = "{dir}/config.json".format(dir=transient)

    output:
        report = "{dir}/nmt-dir-crawl/evaluation/report".format(dir=transient)

    priority: 50
    run:
        trainData = config["initCorpusTrainPrefix"]

        crawlPref = "{dir}/crawl".format(dir=permanent)
        trainData.append(crawlPref)

        cmd = "snakemake --snakefile {BITEXTOR}/snakemake/nmt/Snakefile --configfile {input.config} -k -j3" \
            + " --directory {transient}/nmt-dir-crawl" \
            + " --config initCorpusTrainPrefix=\"" + str(trainData) + "\"" \
            + " permanentDir={permanent}/nmt-dir-crawl"
        print("cmd", cmd)
        shell(cmd)

rule train_nmt_all:
    input:
        config = "{transient}/config.json"

    output:
        report = "{transient}/nmt-dir/evaluation/report"

    priority: 50
    run:
        cmd = "snakemake --snakefile {BITEXTOR}/snakemake/nmt/Snakefile --configfile {input.config} -k -j3" \
            + " --directory {transient}/nmt-dir" \
            + " --config permanentDir={permanent}/nmt-dir"
        shell(cmd)

rule create_config:
    output:
        config = temp("{dir}/config.json".format(dir=transient))

    priority: 50
    run:
        with open(output.config, "wt") as configFile:
            configFile.write(str(config))


#================================== CRAWLING ======================================#
rule creepy_download:
    params:
        url="http://{target}"
    output:
        '{dir}/warc'.format(dir=permanent)+'/{target}/creepy.warc.xz'
    priority: 10
    shell:
        #'echo {params.url}; '
        '{PROFILING} python3 {BITEXTOR}/bitextor-creepy.py {TLD_CRAWL} {CRAWLSIZELIMIT} {CRAWLTIMELIMIT} {CRAWLWAIT} {CRAWLJOBS} {CRAWLTIMEOUT} {CRAWLDUMPARGS} {CONTINUECRAWL} {USERAGENT} {params.url} | xz -c -T 0 > {output}'

rule httrack_download:
    output:
        '{dir}/warc'.format(dir=permanent)+'/{target}/httrack.warc.xz'
    params:
        url="http://{target}"
    priority: 10
    shell:
        'echo hostname=$HOSTNAME; '
        #'mkdir -p {permanent}/warc/{wildcards.target} ; '
        'DIRNAME=$(mktemp -d {TMPDIR}/downloaded.{wildcards.target}.XXXXXX); '
        '{PROFILING} nice ionice -c 3 {BITEXTOR}/bitextor-httrack.py --url {params.url} --output-path $DIRNAME {CRAWLTIMELIMIT} {CRAWLPAGELIMIT} {USERAGENT} {CRAWLWAIT}; '
        '{PROFILING} nice ionice -c 3 {BITEXTOR}/bitextor-webdir2warc.sh $DIRNAME | nice ionice -c 3 xz -c -T 0 > {output}; '
        #'rm -rf $DIRNAME;'

rule wget_download:
    output:
        '{dir}/warc'.format(dir=permanent)+'/{target}/wget.warc.xz'
    params:
        url="http://{target}"
    priority: 10
    shell:
        'echo hostname=$HOSTNAME; '
        'DIRNAME=$(mktemp -d "{TMPDIR}/downloaded.{wildcards.target}.XXXXXX"); '
        'WGETWARCPATH=$DIRNAME/output;'
        '{PROFILING} nice ionice -c 3 {BITEXTOR}/bitextor-wget.py --url {params.url} --output-path "$DIRNAME" {CRAWLTIMELIMIT} {USERAGENT} {CRAWLFILETYPES} {CRAWLWAIT} --warc $WGETWARCPATH; '
        'cat $WGETWARCPATH.warc | nice ionice -c 3 xz -c -T 0 > {output};'
        'rm -rf $DIRNAME;'


def GetDomainHosts(wildcards):
    output=[]
    for h in domainKey2Hosts[wildcards.target]:
        output.append('{dir}/warc/{host}/{crawler}.warc.xz'.format(dir=permanent, host=h, crawler=CRAWLTARGET))
    return output

rule concat_subdomains:
    input:
        GetDomainHosts
    output:
        "{dir}".format(dir=transient)+"/{target}/concat.warc.xz"
    priority: 9
    run:
        assert(len(input))
        if len(input) == 1:
            cmd = "ln -sfn {input} {output}; "
        else:
            cmd = 'xzcat -T 0 {input} -f | xz -c -T 0 > {output}; '
        shell(cmd)

rule warc2preprocess:
    input:
        '{dir}/concat.warc.xz'
    output:
        deboil='{dir}/deboilerplate_html.xz',
        encoding='{dir}/encoding.xz',
        lang='{dir}/lang.xz',
        mime='{dir}/mime.xz',
        html='{dir}/normalized_html.xz',
        text='{dir}/plain_text.xz',
        url='{dir}/url.xz'
    priority: 8
    shell:
        #By removing the checkpoint every time, we ensure that rule split_warc will be re-run every time, which allows to resume the pipeline if this step breaks
        '{PROFILING} nice ionice -c 3 {BITEXTOR}/bitextor-warc2preprocess.py {boilerpipeCleaning} --output-dir {wildcards.dir} --lang1 {LANG1} --lang2 {LANG2} --input {input} {usepdfextract}; '
        'if [ "{boilerpipeCleaning}" == "" ]; then ln -sfn {output.html} {output.deboil}; fi'

#================================== DICTIONARY-BASED DOCUMENT ALIGNMENT ==================================#
rule lettr2idx:
    input:
        text='{dir}/plain_text.xz',
        lang='{dir}/lang.xz',
    output:
        '{dir}/idx.xz'
    shell:
        '{PROFILING} {BITEXTOR}/bitextor-buildidx.py  --lang1 {LANG1} --lang2 {LANG2} --wordtokeniser1 "{WORDTOK1}" --wordtokeniser2 "{WORDTOK2}" -m 15 --lang {input.lang} --text {input.text} | xz -T 0 > {output}'

rule idx2ridx_l1tol2:
    input:
        '{dir}/idx.xz',
        expand("{dic}", dic=DIC)
    output:
        '{dir}/1.ridx.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/bitextor-idx2ridx.py -d {input[1]} --lang1 {LANG1} --lang2 {LANG2} | xz -T 0 > {output}'

rule idx2ridx_l2tol1:
    input:
        '{dir}/idx.xz',
        expand("{dic}", dic=DIC)
    output:
        '{dir}/2.ridx.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/bitextor-idx2ridx.py -d {input[1]} --lang1 {LANG2} --lang2 {LANG1} | xz -T 0 > {output}'

rule ridx2imagesetoverlap:
    input:
        '{dir}/{num}.ridx.xz',
        '{dir}/deboilerplate_html.xz'
    output:
        '{dir}/{num}.imgoverlap.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/features/bitextor-imagesetoverlap.py --html {input[1]} | xz -T 0 > {output}'

rule imagesetoverlap2structuredistance:
    input:
        '{dir}/{num}.imgoverlap.xz',
        '{dir}/deboilerplate_html.xz'
    output:
        '{dir}/{num}.structuredistance.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/features/bitextor-structuredistance.py --html {input[1]} | xz -T 0 > {output}'

rule structuredistance2urldistance:
    input:
        '{dir}/{num}.structuredistance.xz',
        '{dir}/deboilerplate_html.xz',
        '{dir}/url.xz'
    output:
        '{dir}/{num}.urldistance.xz'
    priority: 8
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/features/bitextor-urlsdistance.py --html {input[1]} --url {input[2]} | xz -T 0 > {output}'

rule urldistance2mutuallylinked:
    input:
        '{dir}/{num}.urldistance.xz',
        '{dir}/deboilerplate_html.xz',
        '{dir}/url.xz'
    output:
        '{dir}/{num}.mutuallylinked.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/features/bitextor-mutuallylinked.py --html {input[1]} --url {input[2]} | xz -T 0 > {output}'

rule mutuallylinked2urlscomparison:
    input:
        '{dir}/{num}.mutuallylinked.xz',
        '{dir}/url.xz'
    output:
        '{dir}/{num}.urlscomparison.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/features/bitextor-urlscomparison.py --url {input[1]} | xz -T 0 > {output}'

rule urlscomparison2urlsoverlap:
    input:
        '{dir}/{num}.urlscomparison.xz',
        '{dir}/deboilerplate_html.xz'
    output:
        '{dir}/{num}.urlsoverlap.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/features/bitextor-urlsetoverlap.py --html {input[1]} | xz -T 0 > {output}'

rule urlsoverlap2rank:
    input:
        '{dir}/{num}.urlsoverlap.xz'
    output:
        '{dir}/{num}.rank.xz'
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} {BITEXTOR}/bitextor-rank.py -m {BITEXTOR}/model/keras.model -w {BITEXTOR}/model/keras.weights | xz -T 0 > {output}'

rule aligndocumentsBitextor:
    input:
        '{dir}/1.rank.xz',
        '{dir}/2.rank.xz',
        '{dir}/plain_text.xz',
        '{dir}/url.xz'
    output:
        '{dir}/docalign.bitextor.xz'
    shell:
        '{PROFILING} {BITEXTOR}/bitextor-align-documents.py --text {input[2]} --url {input[3]} -n 1 -i converge -r /dev/null {input[0]} {input[1]} | xz -T 0 > {output}'



################# MT-BASED DOCUMENT ALIGNMENT #################

rule docaling_extracted:
    input:
        text='{dir}/plain_text.xz',
        lang='{dir}/lang.xz',
        url='{dir}/url.xz'
    output:
        "{dir}/docalign/"+"{l1}.extracted.xz".format(l1=LANG1),
        "{dir}/docalign/"+"{l2}.extracted.xz".format(l2=LANG2)
    shell:
        'mkdir -p {wildcards.dir}/docalign; '
        '{PROFILING} {BITEXTOR}/document-aligner/utils/extract_lett.py -x --langs {LANG1},{LANG2} --plaintextfile {input.text} --urlfile {input.url} --langfile {input.lang} --splitter "{SENTTOK1}" --prune_type "words" --prune 80 --output_dir {wildcards.dir}/docalign'

rule docaling_deduped:
    input:
        "{dir}/{lang}.extracted.xz"
    output:
        temp("{dir}/{lang}.extracted.deduped.xz")
    shell:
        "xzcat -T 0 {input}  | cut -d$'\t' -f 2 | sort -T {TMPDIR} --compress-program=gzip | uniq | xz -c -T 0 > {output}"

rule docaling_translate_nmt:
    input:
        source = "{dir}/{prefix}/extracted.deduped.xz"
        ,
        configFile = "{dir}/config.json".format(dir=transient)
        ,
        report = "{transient}/nmt-dir/evaluation/report".format(transient=transient)

    output:
        temp("{dir}/{prefix}.nmt.extracted.deduped.translated.xz")

    priority: 40

    shell:
        'xzcat -T 0 {input.source} | {PROFILING} {BITEXTOR}/snakemake/nmt/translate.sh {input.configFile} {wildcards.dir} {permanent}/nmt-dir | xz -c -T 0 > {output}; '
        'if [ "$(xzcat -T 0 {input} | wc -l)" -ne "$(xzcat -T 0 {output} | wc -l)" ]; then >&2 echo "TRANSLATION ERROR (command {MT_COMMAND}): {input} and {output} should have the same number of lines"; exit 2; fi'

rule docaling_translate_smt:
    input:
        source = "{dir}/{prefix}.extracted.deduped.xz"
        ,
        report = "{transient}/smt-dir/steps/1/REPORTING_report.1.DONE".format(transient=transient)

    output:
        temp("{dir}/{prefix}.smt.extracted.deduped.translated.xz")

    params:
        smtDir = "{0}/smt-dir".format(config["transientDir"])

    shell:
        'xzcat -T 0 {input.source} | {PROFILING} {BITEXTOR}/snakemake/translate-smt.sh {LANG1} {MOSESDIR} {params.smtDir} | xz -c -T 0 > {output}; '
        'if [ "$(xzcat -T 0 {input.source} | wc -l)" -ne "$(xzcat -T 0 {output} | wc -l)" ]; then >&2 echo "TRANSLATION ERROR (command {MT_COMMAND}): {input.source} and {output} should have the same number of lines"; exit 2; fi'


rule docaling_custom_translate:
    input:
        "{prefix}.extracted.deduped.xz",
    output:
        temp("{prefix}.customMT.extracted.deduped.translated.xz")
    shell:
        'xzcat -T 0 {input} | {MT_COMMAND} | xz -c -T 0 > {output}; '
        'if [ "$(xzcat -T 0 {input} | wc -l)" -ne "$(xzcat -T 0 {output} | wc -l)" ]; then >&2 echo "TRANSLATION ERROR (command {MT_COMMAND}): {input} and {output} should have the same number of lines"; exit 2; fi'


rule docaling_substitute_translated:
    input:
        extracted="{prefix}.extracted.xz",
        deduped="{prefix}.extracted.deduped.xz",
        translated="{prefix}.{mttype}.extracted.deduped.translated.xz"
    output:
        "{prefix}.{mttype}.extracted.translated.xz"
    shell:
        'xzcat -T 0 {input.extracted} | {PROFILING} python3 {BITEXTOR}/document-aligner/substitute_translated.py --deduplicated {input.deduped} --translated {input.translated} | xz -c -T 0 > {output}'

rule docaling_matches:
    input:
        l1="{dir}/"+"{l1}".format(l1=LANG1)+".{mttype}.extracted.translated.xz",
        l2="{dir}/"+"{l2}.extracted.xz".format(l2=LANG2)
    output:
        "{dir}/"+"{l1}-{l2}".format(l1=LANG1,l2=LANG2)+".{mttype}.matches"
    shell:
        "{PROFILING} python3 {BITEXTOR}/document-aligner/compute_matches.py --lang1 {input.l1} --lang2 {input.l2} --output_matches {output} --threshold {DOC_THRESHOLD} --word_tokeniser '{WORDTOK1}'"

rule aligndocumentsTrainedNMT:
    input:
        matches="{dir}/docalign/"+"{l1}-{l2}.{mttype}.matches".format(l1=LANG1,l2=LANG2,mttype=DOCALIGNEXT),
        text='{dir}/plain_text.xz',
        url='{dir}/url.xz',
        report="{transientdir}/nmt-dir/evaluation/report".format(transientdir=transient)
    output:
        '{dir}/docalign.nmt.xz'

    shell:
        'mkdir -p {wildcards.dir}/docalign; '
        '{PROFILING} python3 {BITEXTOR}/document-aligner/build_docs.py --matches {input.matches} --plaintext {input.text} --url {input.url} --threshold {DOC_THRESHOLD} | xz -c -T 0 > {output}'

rule aligndocumentsCustomMT:
    input:
        matches="{dir}/docalign/"+"{l1}-{l2}.{mttype}.matches".format(l1=LANG1,l2=LANG2,mttype=DOCALIGNEXT),
        text='{dir}/plain_text.xz',
        url='{dir}/url.xz'
    output:
        '{dir}/docalign.customMT.xz'
    shell:
        '{PROFILING} python3 {BITEXTOR}/document-aligner/build_docs.py --matches {input.matches} --plaintext {input.text} --url {input.url} --threshold {DOC_THRESHOLD} | xz -c -T 0 > {output}'

rule aligndocumentsTrainedSMT:
    input:
        matches="{dir}/docalign/"+"{l1}-{l2}.{mttype}.matches".format(l1=LANG1,l2=LANG2,mttype=DOCALIGNEXT),
        text='{dir}/plain_text.xz',
        url='{dir}/url.xz'
    output:
        '{dir}/docalign.smt.xz'
    shell:
        '{PROFILING} python3 {BITEXTOR}/document-aligner/build_docs.py --matches {input.matches} --plaintext {input.text} --url {input.url} --threshold {DOC_THRESHOLD} | xz -c -T 0 > {output}'

#================================== SEGMENT ALIGNMENT ==================================#

rule hunaligndic:
    input:
        expand("{dic}", dic=DIC)
    output:
        '{dir}/hunalign_dic'.format(dir=transient)
    run:
        with open(output[0], "wt") as outw:
            with open(input[0], "rt") as inr:
                header=inr.readline().strip()
                langs=header.split("\t")
                if langs[0] == LANG1 and langs[1] == LANG2:
                    inverse=True
                else:
                    inverse=False
                for inline in inr:
                    columns=inline.strip().split("\t")
                    if inverse:
                        outw.write(columns[1]+" @ "+columns[0]+"\n")
                    else:
                        outw.write(columns[0]+" @ "+columns[1]+"\n")

rule alignsegments_hunalign:
    input:
        '{transientdir}/hunalign_dic'.format(transientdir=transient),
        "{dir}/docalign."+"{extension}".format(extension=DOCALIGNEXT)+".xz"
    output:
        '{dir}/hunalign.segalign.xz'
    shell:
        'xzcat -T 0 -f {input[1]} | {PROFILING} {BITEXTOR}/bitextor-align-segments.py -d {input[0]} -t {TMPDIR} --lang1 {LANG1} --lang2 {LANG2} --hunalign-dir "{BITEXTOR}/hunalign/src/hunalign" --sent-tokeniser_sl "{SENTTOK1}" --sent-tokeniser_tl "{SENTTOK2}" --word-tokeniser_sl "{WORDTOK1}" --word-tokeniser_tl "{WORDTOK2}" | xz -T 0 > {output}'

rule alignsegments_bleualign:
    input:
        aligned_urls="{dir}/bleualign/align.info.gz"
    output:
        alignments='{dir}/bleualign.segalign.xz'
    run:
        with lzma.open(output.alignments, "wt") as algFile:
            with open_gzip_or_plain(input.aligned_urls) as urlsFile:
                for urlAlg in urlsFile:
                    #print("urlAlg", urlAlg)
                    urlAlg = urlAlg.strip()
                    if len(urlAlg) == 0:
                        continue
                    id=urlAlg.split("\t")[0]
                    urls="\t".join(urlAlg.strip().split("\t")[1:])
                    filename=wildcards.dir+"/bleualign/aligned."+str(id)+".gz"
                    with open_gzip_or_plain(filename) as sentFile:
                        for line in sentFile:
                            algFile.write(urls+"\t"+line)
        os.sync()

rule deferred_documents:
    input:
        "{dir}/normalized_html.xz",
        "{dir}/url.xz"
    output:
        "{dir}/html5lib_plain_text.xz",
        "{dir}/deferred_documents.xz"
    shell:
        'paste <(xzcat {input[0]}) <(xzcat {input[1]}) | {PROFILING} python3 {BITEXTOR}/standoff/deferred-document.py | awk \'{{ print $1 | "xz > {output[0]}"; print $3 | "xz > {output[1]}" }}\''

rule deferred_segments:
    input:
        "{dir}/" + SEGMENTALIGNER + ".segalign.xz",
        "{dir}/html5lib_plain_text.xz",
        "{dir}/url.xz",
        "{dir}/deferred_documents.xz"
    output:
        "{dir}/" + SEGMENTALIGNER + ".deferred.segalign.xz"
    shell:
        'xzcat -T 0 -f {input[0]} | {PROFILING} python3 {BITEXTOR}/standoff/deferred-sentences.py <(paste <(xzcat {input[1]}) <(xzcat {input[2]}) <(xzcat {input[3]})) > {output}'

rule cleansegments:
    input:
        "{dir}/" + SEGMENTALIGNER + DEFERREDSENTENCES + ".segalign.xz"
    output:
        "{dir}/"  + SEGMENTALIGNER + ".segclean.xz"
    shell:
        'xzcat -T 0 -f {input} | {PROFILING} {BITEXTOR}/bitextor-cleantextalign.py -q {MINQUALITY} -m {MAXLINES} -s | xz -T 0 > {output}'

#================================== POST PROCESSING ==================================#


rule restorative:
    input:
        "{prefix}.segclean.xz"
    output:
        "{prefix}.restorative.xz"
    shell:
        'xzcat -T 0 -f {input} | {PROFILING} {BITEXTOR}/clean-restorative.sh {LANG1} {LANG2} | xz -T 0 > {output}'

rule bicleaner:
    input:
        segclean="{prefix}"+".{extension}.xz".format(extension=RESTORATIVE),
        model="{model}".format(model=BICLEANER_CONFIG)
    output:
        "{prefix}.bicleaner.scores.xz"
    shell:
        'slang=$(egrep "source_lang" {BICLEANER_CONFIG} | cut -d " " -f 2); '
        'if [ "$slang" == "{LANG1}" ]; then '
        '  xzcat -T 0 -f {input.segclean} | {PROFILING} python3 {BITEXTOR}/bicleaner/bicleaner/bicleaner_classifier_lite.py -q --threshold {BICLEANER_THRESHOLD} - - {BICLEANER_CONFIG} | xz -T 0 > {output}; '
        'else '
        '  xzcat -T 0 -f {input.segclean} | awk \' BEGIN {{FS="\t"; OFS="\t"}} {{ print $1 "\t" $2 "\t" $4 "\t" $3 "\t" $5}}\' | {PROFILING} python3  {BITEXTOR}/bicleaner/bicleaner/bicleaner_classifier_lite.py -q --threshold {BICLEANER_THRESHOLD} - - {BICLEANER_CONFIG}  | awk \' BEGIN {{FS="\t"; OFS=","}} {{ print $1 "\t" $2 "\t" $4 "\t" $3 "\t" $5 "\t" $6}}\' | xz -T 0 > {output}; '
        'fi'

rule bicleanerfilter:
    input:
        "{prefix}.bicleaner.scores.xz"
    output:
        "{prefix}.bicleaner.xz"
    shell:
        'xzcat -T 0 -f {input} | {PROFILING} {BITEXTOR}/bitextor-filterbicleaner.py --threshold {BICLEANER_THRESHOLD} | xz -T 0 > {output}'

rule elrc:
    input:
        "{prefix}."+"{extension}".format(extension=BICLEANER)+".xz"
    output:
        "{prefix}.elrc.xz"
    shell:
        'xzcat -T 0 -f {input} | {PROFILING} {BITEXTOR}/bitextor-elrc-filtering.py -c "url1,url2,seg1,seg2,hunalign{DEFERREDFIELDS}{BICLEANEROPTION}" -s | xz -T 0 > {output}'

rule raw:
    input:
        expand("{dir}/{webdomain}/{aligner}.{rawopt}.xz", dir=config["transientDir"], webdomain=domainKey2Hosts.keys(), aligner=SEGMENTALIGNER, rawopt=RAWOPTION)
    output:
        "{permanentdir}".format(permanentdir=config["permanentDir"])+"/{l1}-{l2}.raw.xz"
    run:
        #Original command ("xzcat -T 0 {input} | xz -T 0 > {output}") replaced to be able to deal with any number of inputs
        with open(output[0],'wb') as wfd:
            for f in input:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd, 1024*1024*10)

rule sents:
    input:
        expand("{dir}/{webdomain}/{aligner}.{ext}.xz", dir=config["transientDir"], webdomain=domainKey2Hosts.keys(), aligner=SEGMENTALIGNER, ext={ELRCSCORES})
    output:
        "{permanentdir}".format(permanentdir=config["permanentDir"])+"/{l1}-{l2}.sent.xz"
    run:
        #Original command ("xzcat -T 0 {input} | xz -T 0 > {output}") replaced to be able to deal with any number of inputs
        with open(output[0],'wb') as wfd:
            for f in input:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd, 1024*1024*10)

rule tmx:
    input:
        "{permanentdir}".format(permanentdir=config["permanentDir"])+"/{l1}-{l2}.sent.xz"
    output:
        "{permanentdir}".format(permanentdir=config["permanentDir"])+"/{l1}-{l2}.not-deduped.tmx.xz"
    shell:
        "xzcat -T 0 -f {input} | {PROFILING} {BITEXTOR}/bitextor-buildTMX.py --lang1 {LANG1} --lang2 {LANG2} -c url1,url2,seg1,seg2,hunalign{DEFERREDFIELDS}{BICLEANEROPTION}{ELRCFIELDS} | xz -T 0 > {output}"

rule deduped_tmx:
    input:
        "{permanentdir}".format(permanentdir=config["permanentDir"])+"/{l1}-{l2}.sent.xz"
    output:
        "{permanentdir}".format(permanentdir=config["permanentDir"])+"/{l1}-{l2}.deduped.tmx.xz"
    shell:
        "xzcat -T 0 -f {input} | LC_ALL=C sort -t$'\t' {BICLEANER_SORT} -T {TMPDIR} --compress-program=gzip | {PROFILING} {BITEXTOR}/bitextor-buildTMX-dedup.py --lang1 {LANG1} --lang2 {LANG2} -c url1,url2,seg1,seg2,hunalign{BICLEANEROPTION}{ELRCFIELDS}{DEFERREDFIELDS} | xz -T 0 > {output}"

rule mt_parallel_data:
    input:
         "{permanentdir}/{l1}-{l2}.sent".format(permanentdir=permanent, l1=LANG1, l2=LANG2)+".xz"
    output:
         l1="{permanentdir}/crawl.{lang}".format(permanentdir=permanent, lang=LANG1)
         ,
         l2="{permanentdir}/crawl.{lang}".format(permanentdir=permanent, lang=LANG2)
    shell:
         "xzcat -T 0 -f {input} | cut -f 3,4 | sort -T {TMPDIR} --compress-program=gzip | uniq > corpus; "
         "cut -f 1 corpus > {output.l1} && cut -f 2 corpus > {output.l2}; "
         "rm corpus"

################# BLEUALIGN RULES #################

rule bleualign:
    input:
        extracted_l1="{dir}/docalign/"+"{l1}.extracted.xz".format(l1=LANG1),
        extracted_l2="{dir}/docalign/"+"{l2}.extracted.xz".format(l2=LANG2),
        translated="{dir}/docalign/"+"{l1}.{mttype}.extracted.translated.xz".format(l1=LANG1,mttype=DOCALIGNEXT),
        matches="{dir}/docalign/"+"{l1}-{l2}.{mttype}.matches".format(l1=LANG1,l2=LANG2,mttype=DOCALIGNEXT)
    output:
        "{dir}/bleualign/align.info.gz"
    shell:
        "mkdir -p {wildcards.dir}/bleualign; "
        "{PROFILING} {BITEXTOR}/bleualign-cpp/bleualign_cpp --text1 {input.extracted_l1} --text2 {input.extracted_l2} --text1translated {input.translated} --matches {input.matches} --doc-threshold {DOC_THRESHOLD} --bleu-threshold {BLEU_THRESHOLD} --output-dir {wildcards.dir}/bleualign"

#################### TRAIN BILINGUAL DICTIONARIES #############################

#Temporal directories for generated data
preprocCorpusDir=transient+"/tempcorpuspreproc."+LANG1+"-"+LANG2
mgizaModelDir=transient+"/tempgizamodel."+LANG1+"-"+LANG2

#Input data prefixes
if "initCorpusTrainPrefix" in config:
    trainPrefixes=config["initCorpusTrainPrefix"]
else:
    trainPrefixes=None

#Obtaining the harmonic probability of each pair of words in both directions and filtering out those with less than p=0.2; printing the dictionary
rule symmetrise_dic:
    input:
        vcb1="{dir}/corpus.{l1}.filtered.vcb".format(dir=mgizaModelDir, l1=LANG1),
        vcb2="{dir}/corpus.{l2}.filtered.vcb".format(dir=mgizaModelDir, l2=LANG2),
        t3_1="{dir}/corpus.{l1}-{l2}.t3.final".format(dir=mgizaModelDir, l1=LANG1, l2=LANG2),
        t3_2="{dir}/corpus.{l2}-{l1}.t3.final".format(dir=mgizaModelDir, l1=LANG1, l2=LANG2)
    output:
        "{file}".format(file=DIC)
    run:
        svocabulary={}
        tvocabulary={}
        svcb=open(input.vcb1,"r")
        tvcb=open(input.vcb2,"r")
        for line in svcb:
            item=line.strip().split(" ")
            svocabulary[item[0]]=item[1]

        for line in tvcb:
            item=line.strip().split(" ")
            tvocabulary[item[0]]=item[1]

        t3dic={}
        t3s=open(input.t3_1,"r")
        t3t=open(input.t3_2,"r")
        for line in t3t:
            item=line.strip().split(" ")
            if item[1] in t3dic:
                t3dic[item[1]][item[0]]=item[2]
            else:
                t3dic[item[1]]={}
                t3dic[item[1]][item[0]]=item[2]

        dic=open(output[0], "wt")
        dic.write(LANG1+"\t"+LANG2+"\n")
        for line in t3s:
            item=line.strip().split(" ")
            if item[0] in t3dic:
                if item[1] in t3dic[item[0]]:
                    value1=float(t3dic[item[0]][item[1]])
                    value2=float(item[2])
                    hmean=2/((1/value1)+(1/value2))

                    if hmean > 0.1:
                        if item[1] in svocabulary and item[0] in tvocabulary:
                            word1=svocabulary[item[1]]
                            word2=tvocabulary[item[0]]
                            if word1.isalpha() or word2.isalpha():
                                dic.write("{0}\t{1}\n".format(word1, word2))
        svcb.close()
        tvcb.close()
        t3s.close()
        t3t.close()
        dic.close()
        os.sync()
rule filter_dics:
    input:
        "{prefix}.vcb"
    output:
        "{prefix}.filtered.vcb"
    shell:
        "cat {input} | egrep ' [^ ][^ ]+$' > {output}"

rule mgiza:
    input:
        vcb1="{prefix}.{l1}.vcb",
        vcb2="{prefix}.{l2}.vcb",
        snt="{prefix}.{l2}-{l1}-int-train.snt",
        cooc="{prefix}.{l2}-{l1}.cooc"
    output:
        "{prefix}.{l2}-{l1}.t3.final"
    shell:
        "{PROFILING} {BITEXTOR}/mgiza/mgizapp/bin/mgiza -ncpus 8 -CoocurrenceFile {input.cooc} -c {input.snt} -m1 5 -m2 0 -m3 3 -m4 3 -mh 5 -m5 0 -model1dumpfrequency 1 -o {wildcards.prefix}.{wildcards.l2}-{wildcards.l1} -s {input.vcb1} -t {input.vcb2} -emprobforempty 0.0 -probsmooth 1e-7 2> /dev/null > /dev/null"


rule snt2cooc:
    input:
        vcb1="{prefix}.{l1}.vcb",
        vcb2="{prefix}.{l2}.vcb",
        vcb1cls="{prefix}.{l1}.vcb.classes",
        vcb2cls="{prefix}.{l2}.vcb.classes",
        snt="{prefix}.{l2}-{l1}-int-train.snt"
    output:
        "{prefix}.{l2}-{l1}.cooc"
    shell:
        "{PROFILING} {BITEXTOR}/mgiza/mgizapp/bin/snt2cooc {output} {input.vcb1} {input.vcb2} {input.snt} 2> /dev/null"

rule mkcls:
    input:
        "{dir}/corpus.clean.{{lang}}".format(dir=preprocCorpusDir)
    output:
        "{dir}/corpus.{{lang}}.vcb.classes".format(dir=mgizaModelDir)
    priority: 40

    shell:
        "{PROFILING} {BITEXTOR}/clustercat/bin/mkcls -c50 -n2 -p{input} -V{output} opt 2> /dev/null > /dev/null"

rule plain2snt:
    input:
        l1="{dir}/corpus.clean.{l1}".format(dir=preprocCorpusDir, l1=LANG1),
        l2="{dir}/corpus.clean.{l2}".format(dir=preprocCorpusDir, l2=LANG2)
    output:
        snt_2_1="{dir}/corpus.{l2}-{l1}-int-train.snt".format(dir=mgizaModelDir, l1=LANG1, l2=LANG2),
        snt_1_2="{dir}/corpus.{l1}-{l2}-int-train.snt".format(dir=mgizaModelDir, l1=LANG1, l2=LANG2),
        vcb1="{dir}/corpus.{l1}.vcb".format(dir=mgizaModelDir, l1=LANG1),
        vcb2="{dir}/corpus.{l2}.vcb".format(dir=mgizaModelDir, l2=LANG2)
    priority: 40

    shell:
        "mkdir -p {mgizaModelDir}; "
        "{BITEXTOR}/mgiza/mgizapp/bin/plain2snt {input.l1} {input.l2} 2> /dev/null > /dev/null; "
        "mv {preprocCorpusDir}/corpus.clean.{LANG1}_corpus.clean.{LANG2}.snt {output.snt_2_1}; "
        "mv {preprocCorpusDir}/corpus.clean.{LANG2}_corpus.clean.{LANG1}.snt {output.snt_1_2}; "
        "cp {preprocCorpusDir}/corpus.clean.{LANG1}.vcb {output.vcb1}; "
        "cp {preprocCorpusDir}/corpus.clean.{LANG2}.vcb {output.vcb2}; "

#Clean corpus

rule clean:
    input:
        "{prefix}.tok.low."+"{lang1}".format(lang1=LANG1)
        ,
        "{prefix}.tok.low."+"{lang2}".format(lang2=LANG2)
    output:
        "{prefix}.clean."+"{lang1}".format(lang1=LANG1)
        ,
        "{prefix}.clean."+"{lang2}".format(lang2=LANG2)
    shell:
        "{PROFILING} perl {BITEXTOR}/utils/clean-corpus-n.perl {wildcards.prefix}.tok.low {LANG1} {LANG2} {wildcards.prefix}.clean 1 80 {wildcards.prefix}.lines-retained"

#Lowercase corpus

rule lowercase:
    input:
        "{prefix}.tok.{lang}.xz"
    output:
        "{prefix}.tok.low.{lang}"
    shell:
        "xzcat {input} | {PROFILING} {BITEXTOR}/preprocess/moses/tokenizer/lowercase.perl > {output}"

#Tokenize corpus

rule tokenize_file_l1:
    input:
         expand("{dataset}.{lang}.gz", dataset=trainPrefixes, lang=LANG1)
    output:
        "{preprocCorpusDir}/corpus.tok."+"{lang}.xz".format(lang=LANG1)
    shell:
        "mkdir -p {preprocCorpusDir}; "
        "zcat -f {input} | sed \"s/&apos;/'/g\" | sed 's/&quot;/\"/g' | sed 's/&amp;/\&/g' | {WORDTOK1} | xz -T 0 > {output}"

rule tokenize_file_l2:
    input:
         expand("{dataset}.{lang}.gz", dataset=trainPrefixes, lang=LANG2)
    output:
        "{preprocCorpusDir}/corpus.tok."+"{lang}.xz".format(lang=LANG2)
    shell:
        "mkdir -p {preprocCorpusDir}; "
        "zcat -f {input} | sed \"s/&apos;/'/g\" | sed 's/&quot;/\"/g' | sed 's/&amp;/\&/g' | {WORDTOK2} | xz -T 0 > {output}"

if "bicleanerCorpusTrainingPrefix" in config:
    bicleanerTrainPrefixes=config["bicleanerCorpusTrainingPrefix"]
else:
    bicleanerTrainPrefixes=None

rule bicleaner_train_model:
    input:
        corpusl1=expand("{dataset}.{lang}.xz", dataset=bicleanerTrainPrefixes, lang=LANG1),
        corpusl2=expand("{dataset}.{lang}.xz", dataset=bicleanerTrainPrefixes, lang=LANG2),
        t3_1="{dir}/corpus.{l1}-{l2}.t3.final".format(dir=mgizaModelDir, l1=LANG1, l2=LANG2),
        t3_2="{dir}/corpus.{l2}-{l1}.t3.final".format(dir=mgizaModelDir, l1=LANG1, l2=LANG2)
    output:
        "{model}".format(model=BICLEANER_CONFIG)
    priority: 40

    shell:
        "training=$(mktemp {TMPDIR}/train.XXXXXXXX); "
        "paste <(xzcat -f {input.corpusl1}) <(xzcat -f {input.corpusl2}) > $training; "
        "DIR=$(dirname {BICLEANER_CONFIG}); "
        "echo $DIR; "
        "cp {input.t3_1} $DIR/{LANG1}.dic; "
        "cp {input.t3_2} $DIR/{LANG2}.dic; "
        "gzip $DIR/{LANG1}.dic $DIR/{LANG2}.dic; "
        "lines=$(cat $training | wc -l); "
        "trainlines=$(echo \"$lines*4/10\" | bc); "
        "testlines=$(echo \"($lines-2*$trainlines)/2\" | bc); "
        '{PROFILING} python3  {BITEXTOR}/bicleaner/bicleaner/bicleaner_train.py $training -S "{WORDTOK1}" -T "{WORDTOK2}" --treat_oovs --normalize_by_length -s {LANG1} -t {LANG2} -d $DIR/{LANG1}.dic.gz -D $DIR/{LANG2}.dic.gz -c $DIR/{LANG1}-{LANG2}.classifier -g $trainlines -w $trainlines --good_test_examples $testlines --wrong_test_examples $testlines -m {BICLEANER_CONFIG} --classifier_type random_forest; '
        "rm $training"
