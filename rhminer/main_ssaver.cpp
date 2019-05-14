/**
 * rhminer code
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

#if defined(_WIN32_WINNT) && defined(RH_SCREEN_SAVER_MODE)

#include "BuildInfo.h"
#include "ClientManager.h"
#include "MinersLib/Global.h"

#include <scrnsave.h>
#include <commctrl.h>
#include <tlhelp32.h>
#include "resource.h"

#pragma comment(lib, "ComCtl32.lib")
#pragma comment(lib, "Advapi32.lib")


//#define RH_DEBUG_SSAVER
#pragma comment(lib, "scrnsave.lib")	


// -----------------------------------------------------------
// ssaver data
static bool inited = false;
HBITMAP backBMP;
HWND previewHwnd = 0;
U32 uTimer = 0;
HDC backDC = 0;
int backBufferCX, backBufferCY;
HBITMAP coinBitmap = NULL;
HBITMAP coinBitmap2 = NULL;
std::mutex* pixMutex = 0;
char speedInfo[256];
char errorInfo[256];
U32 acceptedShares = 0;
const U32 MaxRoundColor = 12;
COLORREF roundColors[MaxRoundColor + 1];
U32      roundColor = 0;
bool minerIsRunning = false;
extern BOOL   fChildPreview;
extern HWND   hMainWindow;
extern HINSTANCE hMainInstance;
set<U32>* foundNonces = 0;
const int C_NonceGraphic_Small = 0;
const int C_NonceGraphic_Big = 1;
const int C_NonceGraphic_Icon = 2;
const int C_NonceGraphic_Black = 3;
const int C_NonceGraphic_FullBlack = 4;
char* NonceGraphicNames[] = { "Small Dot", "Big Dot", "Icon", "Black", "Full black" };
U32 currentNonceGraphic = 0;
#define RH_SSAVER_CLEARTIMEOUT (5*60*1000)
extern void DisplayGraphics(U32 nonce);

// -----------------------------------------------------------



//----------------------------------------------------------------------------------------------------------------------------------------

LONG GetStringRegKey(HKEY hKey, const string &strValueName, string &strValue, const string &strDefaultValue)
{
    strValue = strDefaultValue;
    CHAR szBuffer[512];
    DWORD dwBufferSize = sizeof(szBuffer);
    ULONG nError;
    nError = RegQueryValueEx(hKey, strValueName.c_str(), 0, NULL, (LPBYTE)szBuffer, &dwBufferSize);
    if (ERROR_SUCCESS == nError)
    {
        strValue = szBuffer;
    }
    else
    {
        LONG setRes = RegSetValueEx(hKey, strValueName.c_str(), 0, REG_SZ, (LPBYTE)strValue.c_str(), (DWORD)strValue.size() + 1);
        if (setRes != ERROR_SUCCESS)
        {
            MessageBox(GetDesktopWindow(), FormatString("Cannot set registry value %s\n", strValueName.c_str()), "Error", MB_ICONERROR | MB_OK);
        }
    }
    return nError;
}

void SetStringRegKey(HKEY hKey, const string &strValueName, string &strValue)
{
    LONG setRes = RegSetValueEx(hKey, strValueName.c_str(), 0, REG_SZ, (LPBYTE)strValue.c_str(), (DWORD)strValue.size() + 1);
    if (setRes != ERROR_SUCCESS)
    {
        MessageBox(GetDesktopWindow(), FormatString("Cannot set registry value %s\n", strValueName.c_str()), "Error", MB_ICONERROR | MB_OK);
    }
}

void WriteRegistryOptions(string serverPort, string user, string cpuCount, U32 nonceGraphic)
{
    HKEY hKey;
    string regPath = "Software\\PascalCoin\\ScreenSaver";
    LONG lRes = RegOpenKeyEx(HKEY_CURRENT_USER, regPath.c_str(), 0, KEY_READ | KEY_WRITE, &hKey);
    if ((lRes == ERROR_SUCCESS) || (lRes == ERROR_FILE_NOT_FOUND))
    {
        auto status = RegCreateKeyExA(HKEY_CURRENT_USER, regPath.c_str(), 0, NULL, REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_QUERY_VALUE, NULL, &hKey, NULL);
        if (status != ERROR_SUCCESS)
        {
            MessageBox(GetDesktopWindow(), "Cannot create registry key\n", "Error", MB_ICONERROR | MB_OK);
            return;
        }
    }

    string ngs = NonceGraphicNames[currentNonceGraphic];
    SetStringRegKey(hKey, "NonceGraphic", ngs);
    SetStringRegKey(hKey, "ServerPort", serverPort);
    SetStringRegKey(hKey, "User", user);
    SetStringRegKey(hKey, "CpuCount", cpuCount);
}

string ReadRegistryOptions(string& serverPort, string& user, string& cpuCount, string& extra, U32& nonceGraphic)
{
    nonceGraphic = 0;
    HKEY hKey;
    string defOpt = "-apiport 0 -v 2 -r 100 -s mine.pool.pascalpool.org:3333 -su 1300378-87.0.Donations -cpu -cputhreads 2 -processpriority 1";
    string baseOpt = "-apiport 0 -processpriority 1 -cpu ";
    string defExtra = "-v 2 -r 100 ";
    string defServerPort = "mine.pool.pascalpool.org:3333";
    string defUSer = "1300378-87.0.Donations";
    string defCpus;

    //Set default threads count to 20% of the CPU
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo);
    U32 defCpuCnt = siSysInfo.dwNumberOfProcessors / 5;
    if (defCpuCnt == 0)
        defCpuCnt = 1;
    defCpus = FormatString("%d", defCpuCnt);

#ifdef RH_DEBUG_SSAVER
    baseOpt += " -logfilename C:\\TEMP\\ssaver_debug.log ";
#endif


    string regPath = "Software\\PascalCoin\\ScreenSaver";
    LONG lRes = RegOpenKeyEx(HKEY_CURRENT_USER, regPath.c_str(), 0, KEY_READ | KEY_WRITE, &hKey);
    if ((lRes == ERROR_SUCCESS) || (lRes == ERROR_FILE_NOT_FOUND))
    {
        auto status = RegCreateKeyExA(HKEY_CURRENT_USER, regPath.c_str(), 0, NULL, REG_OPTION_NON_VOLATILE, KEY_WRITE | KEY_QUERY_VALUE, NULL, &hKey, NULL);
        if (status != ERROR_SUCCESS)
        {
            MessageBox(GetDesktopWindow(), FormatString("Cannot create registry key %d\n", status), "Error", MB_ICONERROR | MB_OK);
            return defOpt;
        }
    }

    string ngs;
    GetStringRegKey(hKey, "ServerPort", serverPort, defServerPort);
    GetStringRegKey(hKey, "User", user, defUSer);
    GetStringRegKey(hKey, "CpuCount", cpuCount, defCpus);
    GetStringRegKey(hKey, "Extra", extra, defExtra);
    GetStringRegKey(hKey, "NonceGraphic", ngs, NonceGraphicNames[0]);

    for (int i = 0; i < RHMINER_ARRAY_COUNT(NonceGraphicNames); i++)
    {
        if (stricmp(ngs.c_str(), NonceGraphicNames[i]) == 0)
        {
            nonceGraphic = i;
            break;
        }
    }
    defOpt = baseOpt + defExtra + " " + "-s " + serverPort + " " + "-su " + user + " " + "-cputhreads " + cpuCount + " " + extra;

    return defOpt;
}

extern "C" BOOL WINAPI ScreenSaverConfigureDialog(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_INITDIALOG:
    {
        SYSTEM_INFO siSysInfo;
        GetSystemInfo(&siSysInfo);
        string server, user, cpuCnt, extra, maxCpu;
        maxCpu = FormatString(" of %d", siSysInfo.dwNumberOfProcessors);

        string cmdLine = ReadRegistryOptions(server, user, cpuCnt, extra, currentNonceGraphic);

        SetDlgItemText(hDlg, IDC_EDIT1, server.c_str());
        SetDlgItemText(hDlg, IDC_EDIT2, user.c_str());
        SetDlgItemText(hDlg, IDC_EDIT3, cpuCnt.c_str());
        SetDlgItemText(hDlg, IDC_EDIT4, maxCpu.c_str());

        HWND combo = GetDlgItem(hDlg, IDC_COMBO1);
        for (int i = 0; i < RHMINER_ARRAY_COUNT(NonceGraphicNames); i++)
            SendMessage(combo, (UINT)CB_ADDSTRING, (WPARAM)0, (LPARAM)NonceGraphicNames[i]);

        SendMessage(combo, (UINT)CB_SETCURSEL, (WPARAM)currentNonceGraphic, (LPARAM)0);

        return TRUE;
    }
    case WM_COMMAND:
        switch ((WORD)wParam)
        {
        case IDOK:
        {
            string server, user, cpuCnt;
            const int MAX_SIZE_STRING = 256;
            server.resize(MAX_SIZE_STRING);
            user.resize(MAX_SIZE_STRING);
            cpuCnt.resize(MAX_SIZE_STRING);
            GetDlgItemText(hDlg, IDC_EDIT1, (char*)server.c_str(), MAX_SIZE_STRING);
            GetDlgItemText(hDlg, IDC_EDIT2, (char*)user.c_str(), MAX_SIZE_STRING);
            GetDlgItemText(hDlg, IDC_EDIT3, (char*)cpuCnt.c_str(), MAX_SIZE_STRING);

            HWND combo = GetDlgItem(hDlg, IDC_COMBO1);
            currentNonceGraphic = (U32)SendMessage(combo, (UINT)CB_GETCURSEL, (WPARAM)0, (LPARAM)0);

            SYSTEM_INFO siSysInfo;
            GetSystemInfo(&siSysInfo);
            if (ToUInt(cpuCnt) > siSysInfo.dwNumberOfProcessors)
                cpuCnt = toString(siSysInfo.dwNumberOfProcessors);

            WriteRegistryOptions(server, user, cpuCnt, currentNonceGraphic);
            EndDialog(hDlg, TRUE);
        }
        return TRUE;

        case IDCANCEL:
            EndDialog(hDlg, FALSE);
            return TRUE;

        default:
            break;
        }

    default:
        break;
    }

    return FALSE;
}


BOOL WINAPI RegisterDialogClasses(HANDLE hInst)
{
    return TRUE;
}



bool FindProcess(PCSTR name)
{
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 process;
    ZeroMemory(&process, sizeof(process));
    process.dwSize = sizeof(process);
    if (Process32First(snapshot, &process))
    {
        do
        {
            if (string(process.szExeFile) == string(name))
                return true;
        } while (Process32Next(snapshot, &process));
    }

    CloseHandle(snapshot);

    return false;
}

void Initialize()
{
    if (!inited)
    {
        inited = true;
        *speedInfo = 0;
        *errorInfo = 0;
        pixMutex = new std::mutex;
        foundNonces = new std::set<U32>;

        memset(roundColors, 255, sizeof(roundColors));
        roundColors[0] = RGB(236, 112, 99);
        roundColors[1] = RGB(231, 76, 60);
        roundColors[2] = RGB(220, 118, 51);
        roundColors[3] = RGB(186, 74, 0);
        roundColors[4] = RGB(235, 152, 78);
        roundColors[5] = RGB(230, 126, 34);
        roundColors[6] = RGB(243, 156, 18);
        roundColors[7] = RGB(241, 196, 15);
        roundColors[8] = RGB(244, 208, 63);
        roundColors[9] = RGB(212, 172, 13);
        roundColors[10] = RGB(214, 137, 16);
        roundColors[11] = RGB(229, 152, 102);

        string s, u, e, c;
        //read info from reg and make a commandline with them
        string cmdLine = ReadRegistryOptions(s, u, c, e, currentNonceGraphic);

        std::vector<string> cmdOptions = GetTokens(cmdLine, " ");
        int argc = (int)cmdOptions.size();
        char** argv = new char*[argc + 1];
        argv[argc] = 0;
        for (int i = 0; i < cmdOptions.size(); i++)
        {
            argv[i] = new char[cmdOptions[i].length() + 1];
            memcpy(argv[i], cmdOptions[i].c_str(), cmdOptions[i].length() + 1);
        }

        extern int main_init(int argc, char** argv);
        main_init(argc, argv);

        RECT r;
        GetWindowRect(hMainWindow, &r);
        if (!fChildPreview)
        {
            //Detect rhminer already running !!!
            if (!FindProcess("rhminer.exe"))
            {
                PrintOut("Screensaver Running \n");
                ClientManager::I().Initialize();
            }
            else
            {
                minerIsRunning = true;
                PrintOut("Screensaver running without mining\n");
            }
        }
        else
        {
            PrintOut("Screensaver preview \n");
        }
    }
    else
        PrintOut("Screensaver alreadyy initialized\n");
}

void DisplayBitmap(U32 x, U32 y, HBITMAP hbmp)
{
    BITMAP bm;
    HDC hdcMem = CreateCompatibleDC(backDC);
    HBITMAP hbmOld = (HBITMAP)SelectObject(hdcMem, hbmp);

    GetObject(hbmp, sizeof(bm), &bm);
    BitBlt(backDC, x, y, bm.bmWidth, bm.bmHeight, hdcMem, 0, 0, SRCCOPY);

    SelectObject(hdcMem, hbmOld);
    DeleteDC(hdcMem);
}

void DisplaySpeed()
{
    if (currentNonceGraphic != C_NonceGraphic_FullBlack)
    {
        SetTextColor(backDC, RGB(0, 200, 0));
        SetBkColor(backDC, RGB(0, 0, 0));
        const char* str = FormatString("%s  Accepted : %d\n", speedInfo, acceptedShares);
        TextOutA(backDC, 20, 20, str, (int)strlen(str));
    }
}

void DisplayError()
{
    if (currentNonceGraphic != C_NonceGraphic_FullBlack)
    {
        SetTextColor(backDC, RGB(0, 0, 0));
        SetBkColor(backDC, RGB(200, 0, 0));
        TextOutA(backDC, 20, backBufferCY - 100, errorInfo, (int)strlen(errorInfo));
    }
}

void DisplayFoundNonce(U32 nonce)
{
    float yratio = backBufferCY / (float)65535;
    float xratio = backBufferCX / (float)65535;
    U32 y = (nonce / 65535) * yratio;
    U32 x = (nonce % 65535) * xratio;
    DisplayBitmap(x - 16, y - 16, (currentNonceGraphic == C_NonceGraphic_Icon) ? coinBitmap2 : coinBitmap);
}

void CleanScreen()
{
    if (backDC)
    {
        HBRUSH oldBrush = (HBRUSH)SelectObject(backDC, GetStockObject(BLACK_BRUSH));
        Rectangle(backDC, 0, 0, backBufferCX, backBufferCY);
        SelectObject(backDC, oldBrush);
    }
    if (foundNonces)
        foundNonces->clear();
}


LRESULT WINAPI ScreenSaverProc(HWND hWnd, UINT message, WPARAM wparam, LPARAM lparam)
{
    // Handles screen saver messages 
    switch (message)
    {
    case WM_CREATE:
    {
        HWND winhwnd = hWnd;
        RECT r;
        GetWindowRect(winhwnd, &r);
        if (fChildPreview)
        {
            backBufferCX = abs(r.right - r.left);
            backBufferCY = abs(r.bottom - r.top);
        }
        else
        {
            backBufferCX = GetSystemMetrics(SM_CXVIRTUALSCREEN);
            backBufferCY = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        }


        HDC winhdc = GetDC(winhwnd);
        backBMP = (HBITMAP)CreateCompatibleBitmap(winhdc, backBufferCX, backBufferCY);
        backDC = CreateCompatibleDC(winhdc);
        ReleaseDC(winhwnd, winhdc);

        SetTextColor(backDC, RGB(0, 200, 0));
        SetBkColor(backDC, RGB(0, 0, 0));

        Initialize();

        if (fChildPreview)
        {
            coinBitmap = LoadBitmap(GetModuleHandle(NULL), MAKEINTRESOURCE(IDB_BITMAP2));

            HWND parentHWnd = (HWND)_atoi64(__argv[2]);
            RECT rect;
            GetClientRect(parentHWnd, &rect);
            int Width = rect.right;
            int Height = rect.bottom;
            LPCTSTR lpszClassName = TEXT("PascalCoinScreenSaverC");
            WNDCLASSEX wc;
            ZeroMemory(&wc, sizeof(wc));
            wc.cbSize = sizeof(wc);
            wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
            wc.lpfnWndProc = (WNDPROC)ScreenSaverProc;
            wc.cbClsExtra = 0;
            wc.cbWndExtra = 0;
            wc.hInstance = hMainInstance;
            wc.hIcon = NULL;
            wc.hCursor = LoadCursor(NULL, IDC_ARROW);
            wc.hbrBackground = NULL;
            wc.lpszMenuName = NULL;
            wc.lpszClassName = lpszClassName;
            RegisterClassEx(&wc);
            previewHwnd = CreateWindow(lpszClassName, TEXT(""), WS_CHILD | WS_VISIBLE, 0, 0, Width, Height, parentHWnd, NULL, (HINSTANCE)hMainInstance, NULL);
        }
        else
        {
            coinBitmap = LoadBitmap(GetModuleHandle(NULL), MAKEINTRESOURCE(IDB_BITMAP1));
            coinBitmap2 = LoadBitmap(GetModuleHandle(NULL), MAKEINTRESOURCE(IDB_BITMAP_PASSIVE));
        }

        SelectObject(backDC, backBMP);

        PrintOut("Screen saver with desktop width:  %d height:  %d", backBufferCX, backBufferCY);

        if (minerIsRunning)
            uTimer = (UINT)SetTimer(hWnd, 1, 50, NULL);
        else
            uTimer = (UINT)SetTimer(hWnd, 1, 100, NULL);
    }
    break;
    case WM_ERASEBKGND:
    {
    }
    break;
    case WM_TIMER:
    {
        pixMutex->lock();

        if (fChildPreview)
        {
            //...
        }
        else
        {
            if (minerIsRunning)
            {
                DisplayGraphics(0);

                static U64 nextClear = 0;
                if (TimeGetMilliSec() > nextClear)
                {
                    CleanScreen();
                    if (nextClear == 0)
                    {
                        if (currentNonceGraphic != C_NonceGraphic_FullBlack)
                        {
                            const char* str = FormatString("RHminer is allready running...\n");
                            TextOutA(backDC, 20, 20, str, (int)strlen(str));
                        }
                    }
                    nextClear = TimeGetMilliSec() + 1 * 60 * 1000;
                }
            }
            else
                DisplaySpeed();

            RECT r;
            GetClientRect(hWnd, &r);
            InvalidateRect(hWnd, &r, false);
        }

        static U32 statecnt=0;
        statecnt++;
        if ((statecnt % 20) == 0)
        {
            //prevent shutdown of ssaver from screen auto-turn-off
            SetThreadExecutionState(ES_DISPLAY_REQUIRED);
        }

        pixMutex->unlock();
    }
    break;
    case WM_DESTROY:
    {
        ReleaseDC(hWnd, backDC);
        PostQuitMessage(0);
    }
    break;
    case WM_PAINT:
    {
        pixMutex->lock();

        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);

        if (fChildPreview)
        {
            RECT rc;

            GetWindowRect(previewHwnd, &rc);
            BITMAP bm;
            HDC winhdc = GetDC(previewHwnd);
            HDC cdc = CreateCompatibleDC(winhdc);
            HDC hdcMem = CreateCompatibleDC(cdc);
            HBITMAP hbmOld = (HBITMAP)SelectObject(hdcMem, coinBitmap);
            GetObject(coinBitmap, sizeof(bm), &bm);
            StretchBlt(winhdc, 0, 0, abs(rc.right - rc.left), abs(rc.bottom - rc.top), hdcMem, 0, 0, bm.bmWidth, bm.bmHeight, SRCCOPY);

            SelectObject(hdcMem, hbmOld);
            DeleteDC(hdcMem);
            ReleaseDC(previewHwnd, winhdc);
            ReleaseDC(previewHwnd, cdc);
        }
        else
        {
            BitBlt(hdc, 0, 0, backBufferCX, backBufferCY, backDC, 0, 0, SRCCOPY);
        }

        EndPaint(hWnd, &ps);

        pixMutex->unlock();
    }
    break;
    default:
    {
        return DefScreenSaverProc(hWnd, message, wparam, lparam);
    }
    }

    return 0;

}

void ScreensaverFoundNonce(U32 nonce)
{
    if (backDC && currentNonceGraphic != C_NonceGraphic_FullBlack)
    {
        pixMutex->lock();

        DisplayFoundNonce(nonce);
        if (foundNonces)
            foundNonces->insert(nonce);


        pixMutex->unlock();
    }
}

void DisplayGraphics(U32 nonce)
{
    U32 x, y;
    if (nonce)
    {
        //map nonce space  to to screen space
        float yratio = backBufferCY / (float)65535;
        float xratio = backBufferCX / (float)65535;
        y = (nonce / 65535) * yratio;
        x = (nonce % 65535) * xratio;
    }
    else
    {
        x = rand32() % backBufferCX;
        y = rand32() % backBufferCY;
    }

    static U32 skipcounter = 0;
    skipcounter++;

    switch (currentNonceGraphic)
    {
    case C_NonceGraphic_Small:
    {
        SetPixel(backDC, x, y, roundColors[roundColor]);
    }
    break;
    case C_NonceGraphic_Big:
    {
        SetPixel(backDC, x, y, roundColors[roundColor]);
        SetPixel(backDC, x + 1, y, roundColors[roundColor]);
        SetPixel(backDC, x, y + 1, roundColors[roundColor]);
        SetPixel(backDC, x + 1, y + 1, roundColors[roundColor]);
    }
    break;
    case C_NonceGraphic_Icon:
    {
        if ((skipcounter % 10) == 0)
        {
            DisplayBitmap(x - 16, y - 16, coinBitmap);

            //redisplay found nonces
            if (foundNonces)
            {
                for (auto n : *foundNonces)
                    DisplayFoundNonce(n);
            }

        }
    }
    break;
    case C_NonceGraphic_Black:
    case C_NonceGraphic_FullBlack:
    {
    }
    break;
    }

}

void ScreensaverFeed(U32 nonce)
{
    if (backDC)
    {
        pixMutex->lock();
        DisplayGraphics(nonce);
        pixMutex->unlock();
    }
}

void ScreenSaverText(const char* szBuffer)
{
    bool isAaccepted = !!strstr(szBuffer, "Share accepted");
    bool isNewWork = !!stristr(szBuffer, "Received new Work");
    bool isSpeed = !!strstr(szBuffer, "Speed :");
    bool isError = !!stristr(szBuffer, "error") || stristr(szBuffer, "Exception");

    if (isAaccepted)
        acceptedShares++;

    if (isNewWork)
    {
        *errorInfo = 0;
        if (backDC)
        {
            pixMutex->lock();

            roundColor = rand32() % MaxRoundColor;

            CleanScreen();

            DisplaySpeed();
            //DisplayError();

            pixMutex->unlock();
        }
    }

    if (isSpeed || isAaccepted)
    {
        if (isSpeed)
        {
            const char* src = szBuffer;
            char* fnd = stristr(szBuffer, "Speed :");
            if (fnd)
                src = fnd;
            strncpy(speedInfo, src, sizeof(speedInfo));
            fnd = strchr(speedInfo, '(');
            if (fnd)
                *fnd = 0;
        }

        if (backDC)
        {
            pixMutex->lock();
            DisplaySpeed();
            pixMutex->unlock();
        }
    }

    if (isError)
    {
        strncpy(errorInfo, szBuffer, sizeof(errorInfo));
        if (backDC)
        {
            pixMutex->lock();
            DisplayError();
            pixMutex->unlock();
        }
    }
}

#endif //winnt&saver
