/**
 * Generic command line parser
 *
 * Copyright 2018 Polyminer1 <https://github.com/polyminer1>
 *
 * To the extent possible under law, the author(s) have dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along with
 * this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

///
/// @file
/// @copyright Polyminer1, QualiaLibre

#include "precomp.h"
#include "CommandLineManager.h"
#include "corelib/CommonData.h"

S32 RHMINER_ValidateGlobalVarRange(const char* varName, const string& valStr, S32 _min, S32 _max)
{
    S32 val = 0;
    if (valStr.find("0x") != string::npos)
        val = ToIntX(valStr);
    else
        val = ToInt(valStr);

    if (_min != _max)
    {
        if (val < _min || val > _max)
        {
            PrintOut("ERROR: Comand line value for -%s is invalid. Must range from %d to %d. Defaulting to %d\n", varName, _min, _max, _min);
            val = _min;
        }
    }
    return val;
}


CmdLineManager& CmdLineManager::GlobalOptions()
{
    static CmdLineManager i;
    return i;
}

CmdLineManager::CmdLineManager()
{
}

void CmdLineManager::LoadGlobals()
{
    if (this != &GlobalOptions() && m_options.size() == 0)
        Merge(GlobalOptions());
}

string FilterSymbol(const string& s, bool& defPResent)
{
    string symbol = s;
    if (s.find('{') != string::npos)
    {
        defPResent = true;
        symbol = s.substr(0, s.find('{') - 1);
    }
    return symbol;
}

void CmdLineManager::RegisterValue(const string& _symbol, const string& cathegory,  const string& descriptor, ValFunc f)
{
    LoadGlobals();
    bool defaultPresent = false;
    string symbol = FilterSymbol(_symbol, defaultPresent);


    RHMINER_ASSERT(Find(symbol) == NULL);
    CmdLineManagerOption x;
    x.cathegory = cathegory;
    x.descr = descriptor;
    x.symbol = symbol;
    x.valSetter = f;
    x.allowDefault = defaultPresent;
    m_options.push_back( x );
}

void CmdLineManager::RegisterValueMultiple(const string& _symbol, const string& cathegory,  const string& descriptor, ValFunc f)
{
    LoadGlobals();
    bool defaultPresent = false;
    string symbol = FilterSymbol(_symbol, defaultPresent);

    RHMINER_ASSERT(Find(symbol) == NULL);
    CmdLineManagerOption x;
    x.cathegory = cathegory;
    x.descr = descriptor;
    x.symbol = symbol;
    x.valSetter = f;
    x.allowMultiples = true;
    x.allowDefault = defaultPresent;
    m_options.push_back( x );
}

void CmdLineManager::RegisterFlag(const string& symbol, const string& cathegory,  const string& descriptor, FlagFunc f)
{
    LoadGlobals();

    RHMINER_ASSERT(Find(symbol) == NULL);
    CmdLineManagerOption x;
    x.descr = descriptor;
    x.cathegory = cathegory;
    x.symbol = symbol;
    x.flagSetter = f;
    m_options.push_back(x);
}

void CmdLineManager::Reset()
{
    for(auto& o : m_options)
        o.parsed = false;
}

bool CmdLineManager::FindSwitch(const string& switchName)
{
    RHMINER_ASSERT(switchName[0] == '-');
    for (U32 i = 1; i < m_argc; i++)
        if ( strncmp(m_argv[i], switchName.c_str(), switchName.length()) == 0 )
            return true;

    return false;
}

void CmdLineManager::Merge(const CmdLineManager& src)
{
    for(auto& o : src.m_options)
    {
        if (Find(o.symbol) == NULL)
            m_options.push_back(o);
    }
}

extern FILE* Logfile;
void CmdLineManager::LoadFromXml(const char* configFile)
{
    U64 endf = GetFileSize(configFile);
    FILE* f = fopen(configFile, "rb");
    if (f)
    {
        string buffer;
        buffer.resize(endf + 1);
        fread(&buffer[0], endf, 1, f);
        buffer[endf] = 0;
        fclose(f);

        try
        {
            Json::Reader reader;
            if (!reader.parse(&buffer[0], m_xmlCommandLineConfig))
            {
                PrintOutCritical("Cannot parse config file %s \n %s", configFile, reader.getFormattedErrorMessages().c_str());
                exit(-299);
            }
            else
            {
                auto x = m_xmlCommandLineConfig.getMemberNames();
                //find logfn file and apply
                Json::Value params = m_xmlCommandLineConfig.get("logfilename", Json::Value::null);
                if (!params.isNull())
                {
                    string logfn = params.asString();
                    if (logfn.length())
                    {
                        if (!Logfile)
                            SetLogFileName(logfn.c_str());
                    }
                }
            }
        }
        catch (std::exception const& _e) 
        {
            PrintOutCritical("xml parsing error in %s : %s\n", configFile, _e.what());
            exit(-299);
        }
    }
    else
    {
        PrintOutCritical("Cannot open config file %s\n", configFile);
        exit(-299);
    }
}

void CmdLineManager::List()
{
    printf("\nrhminer will automatically load config.txt if no commandline are present\n\n");

    strings cat = {"General", "Optimizations", "Gpu", "Network"};
    for (const auto& o : m_options)
    {
        if (std::find(cat.begin(), cat.end(), o.cathegory) == cat.end())
            cat.push_back(o.cathegory);
    }
    for(auto&c : cat)
    {
        if (c == "*")
            continue;

        printf("\n%s options:\n", c.c_str());

        for (const auto& o : m_options)
        {
            if (o.cathegory == c)
            {
                printf("  -%-20s ", o.symbol.c_str());
                strings lines = GetTokens(o.descr, "\n");

                printf("%s\n", lines[0].c_str());
                for(int i=1; i < lines.size(); i++)
                    printf("                        %s\n", lines[i].c_str());
            }
        }
    }
}

bool CmdLineManager::PreParseSymbol(const char* symbol)
{
    RHMINER_ASSERT(m_argv);
    return ParseInternal(symbol, false) == 0;
}

bool CmdLineManager::Parse(const strings& strList, bool exitOnError)
{
    U32 argc = 0;
    char** args = (char**)malloc((strList.size()+3) * sizeof(char*));
    auto AddArgs = [&] (const char* str)
    {
        U32 maxLen = strlen(str);
        RHMINER_ASSERT(maxLen < 511);
        char* wtf = (char*)malloc(512);
        args[argc] = wtf;
        memcpy(args[argc], str, maxLen + 1);
        argc++;
    };

    AddArgs("rhminer.exe");
    for (auto&v : strList)
        AddArgs(v.c_str());
    args[argc] = 0;

    return Parse(argc, args, exitOnError);
}

Json::Value CmdLineManager::m_xmlCommandLineConfig;
int CmdLineManager::m_argc =0;
char** CmdLineManager::m_argv = 0;
string CmdLineManager::m_argslist;
bool CmdLineManager::Parse(int argc, char** argv, bool exitOnError)
{
    m_argc = argc;
    m_argv = argv;
    m_argslist = "";

    for(int i=1; i < argc; i++)
    {
        m_argslist += argv[i];
        m_argslist += " ";
    }
    
    int res = ParseInternal(NULL, exitOnError);
    if (res && exitOnError)
    {
        printf("Invalid command line option '%s'\n", res < argc ? argv[res]:"");
        RHMINER_EXIT_APP("");
    }

	PrintOutSilent("Options: %s\n", m_argslist.c_str());
	
    return res == 0;
}

//return ith symbol that cause error. else return 0
int CmdLineManager::ParseInternal(const char* specificSymbol, bool exitOnError)
{
    if (!m_xmlCommandLineConfig.empty())
        return ParseInternalXML(specificSymbol, exitOnError);
    else
        return ParseInternalCMD(specificSymbol, exitOnError);
}

int CmdLineManager::ParseInternalXML(const char* specificSymbol, bool exitOnError)
{
    int i = 1;
    try
    {
        auto tokenNames = m_xmlCommandLineConfig.getMemberNames();
        for (auto symb : tokenNames)
        {
            if (specificSymbol && symb != specificSymbol)
                continue;

            try
            {
                CmdLineManagerOption* o = Find(symb);
                if (o)
                {
                    if (o->parsed && !o->allowMultiples)
                        continue;

                    string symbVal = m_xmlCommandLineConfig.get(symb, "").asString();
					//PrintOut("XML option %s '%s'\n", symb.c_str(), symbVal.c_str());
                    if (symbVal.length())
                    {
                        if (o->flagSetter)
                        {
                            if (stristr(symbVal.c_str(), "enable") || stristr(symbVal.c_str(), "true"))
                            {
                                o->flagSetter();
                                o->parsed = true;
                                m_argslist += FormatString("-%s ", symb.c_str());
                            }
                        }
                        else
                        {
                            //detect val with no value !
                            o->valSetter(symbVal);
                            o->parsed = true;
                            m_argslist += FormatString("-%s %s ", symb.c_str(), symbVal.c_str());
                        }
                    }
                }
                else
                {
                    if (symb != "gpu" &&
                        symb != "gputhreads" &&
                        symb != "kernelactivewaiting")
                    {
                    printf("Unknown argument '%s' \n", symb.c_str());
                }
            }
            }
            catch (...)
            {
                if (symb.c_str() && symb.length())
                {
                    printf("Invalid argument value for option %s \n", symb.c_str());
                    RHMINER_EXIT_APP("")
                }
                else
                {
                    printf("Invalid argument values\n");
                    RHMINER_EXIT_APP("")
                }
                return 0;
            }
        }
    }
    catch (...)
    {
        printf("Command line argument error\n");
        return 0;
    }

    if (m_argslist.length() && !specificSymbol)
        OverrideArgs(m_argslist);

    return 0;
}

int CmdLineManager::ParseInternalCMD(const char* specificSymbol, bool exitOnError)
{
    int i = 1;
    try
    {
        for (i = 1; i < m_argc; i++)
        {
            if (m_argv[i][0] == '-')
            {
                string symb = m_argv[i]+1;
                
                if (specificSymbol && symb != specificSymbol)
                    continue;

                try
                {
                    if (ProcessSymbol(symb, i))
                    {
                        i++;
                        for (int j = i; j < m_argc; j++)
                        {
                            if (m_argv[j][0] == '-')
                            {
                                i = j-1;
                                break;
                            }
                        }
                    }
                    else
                    {
                        if (exitOnError)
                            return i;
                    }
                }
                catch (...)
                {
                    if (symb.c_str() && symb.length())
                    {
                        printf("Invalid argument value for option -%s \n", symb.c_str());
                        RHMINER_EXIT_APP("")
                    }
                    else
                    {
                        printf("Invalid argument values\n");
                        RHMINER_EXIT_APP("")
                    }
                    return i;
                }
            }
        }
    }
    catch (...)
    {
        printf("Command line argument error\n");
        return i;
    }
    return 0;
}

bool CmdLineManager::ProcessSymbol(const string& symb, int& i)
{
    CmdLineManagerOption* o = Find(symb);
    if (o)
    {
        if (o->parsed && !o->allowMultiples)
            return true;

        if (o->flagSetter)
        {
            o->flagSetter();
            o->parsed = true;
        }
        else
        {
            //detect val with no value !
            if (i + 1 >= m_argc || m_argv[i + 1][0] == '-')
            {
                if (o->allowDefault)
                {
                    if (!o->parsed)
                    {
                        o->valSetter("DEFAULT");
                        o->parsed = true;
                    }
                }
                else
                    throw RH_Exception("Bad param");
            }
            else
            {
                o->valSetter(m_argv[++i]);
                o->parsed = true;
            }
        }
    }
    else
        return false;

    return true;
}

void CmdLineManager::OverrideArgs(const string& newArgs)
{
    strings na = GetTokens(newArgs, " ");
    
    char* arg0 = m_argv[0];
    m_argv = (char**)malloc(sizeof(int*) * 256);
    m_argv[0] = (char*)malloc(strlen(arg0)+ 1);
    strcpy(m_argv[0], arg0);

    m_argslist = newArgs;
    size_t i;
    for (i=0; i < na.size(); i++)
    {
        m_argv[i + 1] = (char*)malloc(na[i].length() + 1);
        strcpy(m_argv[i + 1], &na[i][0]);
    }
    m_argv[i + 1] = 0;
    m_argc = i+1;
}
